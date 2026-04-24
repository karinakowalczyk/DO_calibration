"""
Bayesian parameter estimation using a GP emulator as forward model.

Uncertainty model (diagonal, independent observables):
    Var_total_i = sigma_emulator_i^2 + sigma_obs_i^2

Sampling: all parameters in log-space  phi_i = log(theta_i).
Prior   : marginal KDE on training ensemble (physical space) x hard box
          bounds, with log-Jacobian correction.
Sampler : emcee.EnsembleSampler.

obs_targets / obs_sigmas key conventions
-----------------------------------------
'pca_0', 'pca_1', ...  PCA component i of the AMOC PDF
any key from stat_cols  scalar statistic (e.g. 'avg_waiting_time')
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')


class EmulatorMCMC:
    """
    Parameters
    ----------
    emulator      : fitted GPEmulatorPDF
    stat_cols     : list[str] — names of scalar statistics in emulator output order
    prior_bounds  : dict {param_name: (lo, hi)} — hard bounds in physical space
    obs_targets   : dict {key: value} — calibration targets (see module docstring)
    obs_sigmas    : dict {key: sigma} — observational uncertainties (same keys)
    prior_samples : (n, n_params) array — training parameters used to build KDE prior
    nwalkers      : int
    """

    def __init__(self, emulator, stat_cols, prior_bounds,
                 obs_targets, obs_sigmas, prior_samples, nwalkers=64):
        self.emulator      = emulator
        self.stat_cols     = list(stat_cols)
        self.prior_bounds  = dict(prior_bounds)
        self.obs_targets   = dict(obs_targets)
        self.obs_sigmas    = dict(obs_sigmas)
        self.nwalkers      = nwalkers

        self.param_names = list(prior_bounds.keys())
        self.ndim        = len(self.param_names)

        self.lower = np.array([v[0] for v in prior_bounds.values()])
        self.upper = np.array([v[1] for v in prior_bounds.values()])
        self.log_lower = np.log(self.lower)
        self.log_upper = np.log(self.upper)

        self._prior_kde     = gaussian_kde(prior_samples.T)
        self._prior_samples = prior_samples.copy()

        # Parse likelihood keys
        self._pca_keys  = sorted(
            [k for k in obs_targets if k.startswith('pca_')],
            key=lambda k: int(k.split('_')[1]),
        )
        self._stat_keys = [k for k in obs_targets if not k.startswith('pca_')]
        self._stat_idx  = {k: self.stat_cols.index(k) for k in self._stat_keys}

        self.sampler         = None
        self._samples_phi    = None
        self._samples_theta  = None
        self._burn_in        = None
        self._thin           = None

    # ── coordinate transforms ──────────────────────────────────────────────

    def _phi_to_theta(self, phi):
        return np.exp(np.asarray(phi, dtype=float))

    def _theta_to_phi(self, theta):
        return np.log(np.asarray(theta, dtype=float))

    # ── prior ──────────────────────────────────────────────────────────────

    def log_prior(self, phi):
        phi = np.asarray(phi, dtype=float)
        if not np.all((phi >= self.log_lower) & (phi <= self.log_upper)):
            return -np.inf
        theta   = self._phi_to_theta(phi)
        kde_val = self._prior_kde(theta.reshape(-1, 1))[0]
        if kde_val <= 0.0:
            return -np.inf
        # log p(phi) = log p(theta) + log|J|,  J_ii = exp(phi_i)
        return np.log(kde_val) + np.sum(phi)

    # ── likelihood ─────────────────────────────────────────────────────────

    def log_likelihood(self, phi):
        theta           = self._phi_to_theta(phi).reshape(1, -1)
        pca_p, Y_p, pca_s, Y_s = self.emulator.predict(theta, return_std=True)
        pca_p, pca_s    = pca_p.flatten(), pca_s.flatten()
        Y_p,   Y_s      = Y_p.flatten(),   Y_s.flatten()

        ll = 0.0
        for key in self._pca_keys:
            i   = int(key.split('_')[1])
            var = pca_s[i]**2 + self.obs_sigmas[key]**2
            ll -= 0.5 * ((self.obs_targets[key] - pca_p[i])**2 / var
                         + np.log(2.0 * np.pi * var))

        for key in self._stat_keys:
            j   = self._stat_idx[key]
            var = Y_s[j]**2 + self.obs_sigmas[key]**2
            ll -= 0.5 * ((self.obs_targets[key] - Y_p[j])**2 / var
                         + np.log(2.0 * np.pi * var))
        return ll

    # ── posterior ──────────────────────────────────────────────────────────

    def log_posterior(self, phi):
        lp = self.log_prior(phi)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(phi)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # ── initialisation ─────────────────────────────────────────────────────

    def _init_walkers(self, seed=42):
        rng = np.random.default_rng(seed)
        phi = (self.log_lower
               + (self.log_upper - self.log_lower)
               * rng.random((self.nwalkers, self.ndim)))
        bad = [i for i in range(self.nwalkers)
               if not np.isfinite(self.log_posterior(phi[i]))]
        if bad:
            raise RuntimeError(
                f"{len(bad)} walkers have non-finite posterior at initialisation. "
                "Check prior bounds and KDE coverage."
            )
        return phi

    # ── run ────────────────────────────────────────────────────────────────

    def run(self, n_steps=10000, burn_in=1000, thin=10, seed=42, progress=True):
        """
        Run emcee and store thinned post-burn-in samples.

        Parameters
        ----------
        n_steps  : total chain length per walker
        burn_in  : steps discarded at the start
        thin     : keep every `thin`-th step after burn-in
        """
        import emcee

        phi_init = self._init_walkers(seed=seed)
        print(f"emcee: {self.nwalkers} walkers × {n_steps} steps  "
              f"(burn-in {burn_in}, thin {thin})")
        print(f"Likelihood terms: {self._pca_keys + self._stat_keys}\n")

        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_posterior
        )
        self.sampler.run_mcmc(phi_init, n_steps, progress=progress)

        self._burn_in       = burn_in
        self._thin          = thin
        self._samples_phi   = self.sampler.get_chain(
            discard=burn_in, thin=thin, flat=True
        )
        self._samples_theta = np.exp(self._samples_phi)
        return self

    # ── posterior access ───────────────────────────────────────────────────

    @property
    def samples(self):
        """Posterior samples in physical parameter space, shape (N, ndim)."""
        if self._samples_theta is None:
            raise RuntimeError("Call run() first.")
        return self._samples_theta

    # ── convergence diagnostics ────────────────────────────────────────────

    def convergence_diagnostics(self):
        """Print acceptance fractions and autocorrelation times."""
        acc   = self.sampler.acceptance_fraction
        n_s   = self._samples_phi.shape[0]

        print("=" * 55)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 55)
        print(f"Posterior samples      : {n_s}")
        print(f"Acceptance fraction    : {acc.mean():.3f}  "
              f"[{acc.min():.3f} – {acc.max():.3f}]"
              f"  (ideal: 0.2 – 0.5)")

        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            print(f"\nIntegrated autocorr. time τ (steps):")
            for name, t in zip(self.param_names, tau):
                warn = "  ⚠ chain too short" if n_s < 50 * t else ""
                print(f"  {name:<22}: {t:6.1f}{warn}")
            print(f"\nMean τ                 : {tau.mean():.1f}")
            print(f"Effective samples (N/τ): {n_s / tau.mean():.0f}")
        except Exception as e:
            print(f"\nCould not estimate τ ({e}). "
                  "Chain may be too short.")
        print("=" * 55)

    # ── chain traces ───────────────────────────────────────────────────────

    def plot_chains(self, figsize=None):
        """Walker traces for each parameter (physical space) + log-posterior."""
        chain = self.sampler.get_chain()    # (n_steps, n_walkers, ndim)
        logp  = self.sampler.get_log_prob() # (n_steps, n_walkers)

        n_plots  = self.ndim + 1
        figsize  = figsize or (14, 2.2 * n_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            ax.plot(np.exp(chain[:, :, i]), color='steelblue', alpha=0.1, lw=0.4)
            ax.set_ylabel(name, fontsize=8)
            ax.axvline(self._burn_in, color='red', ls='--', lw=1, alpha=0.8)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

        axes[-1].plot(logp, color='darkorange', alpha=0.1, lw=0.4)
        axes[-1].set_ylabel('log P', fontsize=8)
        axes[-1].axvline(self._burn_in, color='red', ls='--', lw=1,
                         label=f'burn-in ({self._burn_in})')
        axes[-1].legend(fontsize=8, loc='upper right')
        axes[-1].set_xlabel('Step')
        axes[-1].grid(alpha=0.2)
        axes[-1].tick_params(labelsize=7)

        plt.suptitle('MCMC Chains', fontsize=12)
        plt.tight_layout()
        return fig

    # ── autocorrelation time vs chain length ───────────────────────────────

    def plot_autocorr(self, figsize=(8, 4)):
        """Integrated τ vs chain length — shows whether the chain has converged."""
        import emcee

        chain   = self.sampler.get_chain()
        n_steps = chain.shape[0]
        start   = max(50, self.ndim * 10)
        checkpoints = np.unique(
            np.logspace(np.log10(start), np.log10(n_steps), 25).astype(int)
        )

        taus = []
        for n in checkpoints:
            try:
                t = emcee.autocorr.integrated_time(chain[:n], quiet=True).mean()
            except Exception:
                t = np.nan
            taus.append(t)
        taus = np.array(taus)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(checkpoints, taus, 'o-', color='steelblue', ms=4, label='Mean τ')
        ax.plot(checkpoints, checkpoints / 50, 'k--', lw=1,
                label='N / 50  (convergence criterion)')
        ax.axvline(self._burn_in, color='red', ls='--', lw=1, label='burn-in')
        ax.set_xlabel('Chain length')
        ax.set_ylabel('Integrated autocorr. time τ')
        ax.set_title('Autocorrelation time vs chain length\n'
                     '(converged when τ curve lies below N/50 line)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    # ── acceptance fractions ───────────────────────────────────────────────

    def plot_acceptance(self, figsize=(10, 3)):
        """Per-walker acceptance fractions."""
        acc = self.sampler.acceptance_fraction
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(np.arange(self.nwalkers), acc,
               color='steelblue', alpha=0.75, width=0.9)
        ax.axhline(acc.mean(), color='crimson', ls='--', lw=2,
                   label=f'Mean = {acc.mean():.3f}')
        ax.axhspan(0.2, 0.5, alpha=0.12, color='green',
                   label='Target range  0.2 – 0.5')
        ax.set_xlabel('Walker index')
        ax.set_ylabel('Acceptance fraction')
        ax.set_title('Per-walker acceptance fractions')
        ax.set_xlim(-0.5, self.nwalkers - 0.5)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    # ── 1-D marginal posteriors ────────────────────────────────────────────

    def plot_posterior_1d(self, figsize=None, bins=40):
        """
        1-D marginal posterior (histogram) + prior KDE (from training samples)
        for each parameter. Median and 68 % credible interval annotated.
        """
        n_cols = 3
        n_rows = int(np.ceil(self.ndim / n_cols))
        figsize = figsize or (5 * n_cols, 3.5 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i, (ax, name) in enumerate(zip(axes, self.param_names)):
            lo, hi = self.lower[i], self.upper[i]
            grid   = np.linspace(lo, hi, 300)

            # Marginal prior KDE from training samples (1-D projection)
            prior_k = gaussian_kde(self._prior_samples[:, i])
            prior_d = prior_k(grid)
            prior_d /= np.trapezoid(prior_d, grid)
            ax.fill_between(grid, prior_d, alpha=0.3, color='gray', label='Prior')
            ax.plot(grid, prior_d, color='gray', lw=1)

            # Posterior histogram
            ax.hist(self.samples[:, i], bins=bins, density=True,
                    alpha=0.7, color='steelblue', label='Posterior', zorder=2)

            # Median + 68 % CI
            q16, med, q84 = np.percentile(self.samples[:, i], [16, 50, 84])
            ax.axvline(med, color='steelblue', lw=2)
            ax.axvspan(q16, q84, alpha=0.2, color='steelblue')

            # Prior bounds
            ax.axvline(lo, color='dimgray', ls=':', lw=1)
            ax.axvline(hi, color='dimgray', ls=':', lw=1)

            ax.set_title(f'{name}\n{med:.3g}  [{q16:.3g}, {q84:.3g}]',
                         fontsize=9)
            ax.set_xlabel(name, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        for j in range(self.ndim, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Marginal Posteriors with Prior', fontsize=13)
        plt.tight_layout()
        return fig

    # ── corner plot ────────────────────────────────────────────────────────

    def plot_corner(self, **kwargs):
        """Joint posterior corner plot (requires the `corner` package)."""
        import corner
        fig = corner.corner(
            self.samples,
            labels=self.param_names,
            show_titles=True,
            title_fmt='.3g',
            quantiles=[0.16, 0.5, 0.84],
            **kwargs,
        )
        return fig

    # ── posterior predictive ───────────────────────────────────────────────

    def posterior_predictive(self, n_samples=100, seed=0):
        """
        Draw n_samples from the posterior and evaluate the emulator.

        Returns
        -------
        pred_pdfs  : (n_samples, n_grid)
        pred_stats : (n_samples, n_stat_cols)  ordered by stat_cols
        pred_pca   : (n_samples, n_pca)
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.samples),
                         size=min(n_samples, len(self.samples)), replace=False)

        pred_pdfs, pred_stats, pred_pca = [], [], []
        for theta in self.samples[idx]:
            pca_p, Y_p = self.emulator.predict(
                theta.reshape(1, -1), return_std=False
            )
            pred_pdfs.append(self.emulator.reconstruct_pdf(pca_p).flatten())
            pred_stats.append(Y_p.flatten())
            pred_pca.append(pca_p.flatten())

        return np.array(pred_pdfs), np.array(pred_stats), np.array(pred_pca)

    def plot_predictive_pdf(self, pred_pdfs, x_grid, target_pdf=None,
                            figsize=(8, 5)):
        """
        Posterior predictive PDF envelope vs the calibration target PDF.

        Parameters
        ----------
        pred_pdfs  : (n_samples, n_grid) from posterior_predictive()
        x_grid     : 1-D AMOC grid matching pred_pdfs columns
        target_pdf : optional 1-D array — observed/default-run PDF
        """
        mean_pdf        = pred_pdfs.mean(0)
        q5,  q95        = np.percentile(pred_pdfs, [ 5, 95], axis=0)
        q25, q75        = np.percentile(pred_pdfs, [25, 75], axis=0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(x_grid, q5,  q95,  alpha=0.2,  color='steelblue',
                        label='5–95th pct')
        ax.fill_between(x_grid, q25, q75,  alpha=0.35, color='steelblue',
                        label='25–75th pct')
        ax.plot(x_grid, mean_pdf, color='steelblue', lw=2,
                label='Posterior mean')
        if target_pdf is not None:
            ax.plot(x_grid, target_pdf, color='crimson', lw=2, ls='--',
                    label='Target (default run)')
        ax.set_xlabel('AMOC (Sv)')
        ax.set_ylabel('Density')
        ax.set_title('Posterior predictive PDF')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_predictive_stats(self, pred_stats, figsize=None):
        """
        Histogram of posterior-predictive scalar statistics vs calibration
        targets (± observational uncertainty shaded).
        """
        keys   = self._stat_keys
        n      = len(keys)
        n_cols = min(n, 3)
        n_rows = int(np.ceil(n / n_cols))
        figsize = figsize or (5 * n_cols, 3.5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, keys):
            j      = self._stat_idx[key]
            vals   = pred_stats[:, j]
            target = self.obs_targets[key]
            sigma  = self.obs_sigmas.get(key)

            ax.hist(vals, bins=30, density=True, alpha=0.75,
                    color='steelblue', label='Posterior predictive')
            ax.axvline(target, color='crimson', lw=2, ls='--',
                       label=f'Target: {target:.2f}')
            if sigma is not None:
                ax.axvspan(target - sigma, target + sigma,
                           alpha=0.15, color='crimson',
                           label=f'±1σ obs ({sigma:.1f})')
            ax.set_xlabel(key)
            ax.set_ylabel('Density')
            ax.set_title(key)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Posterior Predictive Checks — Scalar Statistics',
                     fontsize=13)
        plt.tight_layout()
        return fig
