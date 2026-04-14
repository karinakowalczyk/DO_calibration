# =============================================================================
# MCMC FOLLOWING ELSAESSER ET AL. (2025) - Equation 2
# =============================================================================

import emcee

class MCMCCalibration:
    """
    MCMC calibration following Elsaesser et al. (2025).
    
    Key features:
    - Likelihood based on penalty metrics (Equation 2)
    - Tolerance inflation based on 5th percentile of PPE
    - Uniform prior within parameter bounds
    """
    
    def __init__(self, emulator, param_bounds, penalty_tolerances):
        """
        Parameters
        ----------
        emulator : NNEnsembleEmulator
            Trained emulator predicting penalty metrics
        param_bounds : dict
            Parameter bounds {name: (lower, upper)}
        penalty_tolerances : array
            Tolerance values for each penalty metric (C_tol in Eq. 2)
        """
        self.emulator = emulator
        self.param_names = list(param_bounds.keys())
        self.lower_bounds = np.array([param_bounds[p][0] for p in self.param_names])
        self.upper_bounds = np.array([param_bounds[p][1] for p in self.param_names])
        self.ndim = len(self.param_names)
        
        # Target penalties are 0 (we want model within obs bounds)
        self.penalty_targets = np.zeros(len(penalty_tolerances))
        self.penalty_tolerances = penalty_tolerances
        
        # Variance = (0.5 * tolerance)^2 per Equation 2
        self.penalty_variances = (0.5 * penalty_tolerances) ** 2
        
    def log_prior(self, theta):
        """Uniform prior within bounds."""
        if np.all((theta >= self.lower_bounds) & (theta <= self.upper_bounds)):
            return 0.0
        return -np.inf
    
    def log_likelihood(self, theta):
        """
        Likelihood following Equation 2: L(C_M) = N(C_target, 0.5*C_tol)
        """
        # Predict penalties
        theta_2d = theta.reshape(1, -1)
        penalties_pred = self.emulator.predict(theta_2d, return_std=False).flatten()
        
        # Multivariate normal log-likelihood (diagonal covariance)
        residuals = penalties_pred - self.penalty_targets
        
        log_like = -0.5 * np.sum(
            residuals**2 / self.penalty_variances + 
            np.log(2 * np.pi * self.penalty_variances)
        )
        
        return log_like
    
    def log_posterior(self, theta):
        """Log posterior = log prior + log likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def compute_tolerances_from_ppe(self, penalty_matrix, percentile=5):
        """
        Compute tolerance inflation based on PPE (Section 4 of paper).
        
        Uses 5th percentile of PPE penalties to account for structural error.
        """
        # For each penalty metric, find the 5th percentile
        tolerances = np.percentile(penalty_matrix, percentile, axis=0)
        
        # Ensure minimum tolerance
        tolerances = np.maximum(tolerances, 1e-6)
        
        return tolerances
    
    def run_mcmc(self, nwalkers=64, nsteps=5000, burn_in=1000, thin=10, progress=True):
        """
        Run MCMC sampling.
        
        Returns
        -------
        samples : array, shape (n_samples, ndim)
            Posterior samples in physical parameter space
        sampler : emcee.EnsembleSampler
            Full sampler object for diagnostics
        """
        # Initialize walkers uniformly within bounds
        p0 = np.random.uniform(
            self.lower_bounds, 
            self.upper_bounds,
            size=(nwalkers, self.ndim)
        )
        
        # Verify all initial positions have finite posterior
        for i in range(nwalkers):
            if not np.isfinite(self.log_posterior(p0[i])):
                raise RuntimeError(f"Walker {i} initialized with non-finite posterior")
        
        print(f"Running MCMC: {nwalkers} walkers, {nsteps} steps")
        
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(p0, nsteps, progress=progress)
        
        # Extract samples
        samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        
        # Diagnostics
        acceptance = sampler.acceptance_fraction
        print(f"\nAcceptance fraction: {np.mean(acceptance):.3f} "
              f"[{np.min(acceptance):.3f}, {np.max(acceptance):.3f}]")
        
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"Autocorrelation time: {np.mean(tau):.1f}")
            print(f"Effective samples: {samples.shape[0] / np.mean(tau):.0f}")
        except:
            print("Could not estimate autocorrelation time")
        
        return samples, sampler


# =============================================================================
# DIAGNOSTIC PLOTS (Following Figures 2, 4, 9, 10 in paper)
# =============================================================================

def plot_parameter_output_correlations(X, penalties, param_names, output_names,
                                        title="Parameter-Output Correlations (PPE)",
                                        save_path=None):
    """
    Heat map of parameter-output correlations (Figure 2a, 9a/b in paper).
    """
    n_params = X.shape[1]
    n_outputs = penalties.shape[1]
    
    # Compute correlations
    correlations = np.zeros((n_outputs, n_params))
    
    for i in range(n_outputs):
        for j in range(n_params):
            correlations[i, j] = np.corrcoef(penalties[:, i], X[:, j])[0, 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12)
    
    # Labels
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_outputs))
    ax.set_yticklabels(output_names, fontsize=10)
    
    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Penalty Metrics', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Mark significant correlations (|r| > 0.3)
    for i in range(n_outputs):
        for j in range(n_params):
            if np.abs(correlations[i, j]) > 0.3:
                ax.plot(j, i, 'ko', markersize=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return correlations


def plot_emulator_validation(y_true, y_pred, output_names, save_path=None):
    """
    Emulator validation plots (Figure 4 in paper).
    """
    n_outputs = y_true.shape[1]
    
    n_cols = 4
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(n_outputs):
        ax = axes[i]
        
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Metrics
        r2 = r2_score(true_vals, pred_vals)
        
        ax.set_xlabel('Actual E3 Output', fontsize=10)
        ax.set_ylabel('NN Emulator Output', fontsize=10)
        ax.set_title(f'{output_names[i]}\nR² = {r2:.2f}', fontsize=11)
        ax.grid(alpha=0.3)
    
    # Hide unused
    for i in range(n_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('NN Emulator vs Actual Model Output (cf. Figure 4)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_posterior_histograms(samples_with_obs_unc, samples_without_obs_unc,
                               param_names, param_bounds, default_values=None,
                               save_path=None):
    """
    Posterior parameter histograms (Figure 10 in paper).
    
    Yellow: with observational uncertainty
    Blue: without observational uncertainty
    """
    n_params = len(param_names)
    
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        
        # Histograms
        ax.hist(samples_with_obs_unc[:, i], bins=30, alpha=0.7, 
                color='gold', label='With obs uncertainty', density=True)
        
        if samples_without_obs_unc is not None:
            ax.hist(samples_without_obs_unc[:, i], bins=30, alpha=0.7,
                    color='steelblue', label='Without obs uncertainty', density=True)
        
        # Prior bounds
        low, high = param_bounds[param_names[i]]
        ax.axvline(low, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(high, color='gray', linestyle='--', alpha=0.5)
        
        # Default value if provided
        if default_values is not None and param_names[i] in default_values:
            ax.axvline(default_values[param_names[i]], color='red', 
                      linestyle='-', linewidth=2, label='Default')
        
        ax.set_xlabel(param_names[i], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Posterior Parameter Distributions (cf. Figure 10)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_penalty_landscape(emulator, param_names, param_bounds, 
                           param_idx1, param_idx2, penalty_idx,
                           X_train=None, y_train=None, n_grid=50,
                           save_path=None):
    """
    Plot emulated penalty landscape (Figure 3 in paper).
    """
    # Create grid
    p1_range = np.linspace(param_bounds[param_names[param_idx1]][0],
                           param_bounds[param_names[param_idx1]][1], n_grid)
    p2_range = np.linspace(param_bounds[param_names[param_idx2]][0],
                           param_bounds[param_names[param_idx2]][1], n_grid)
    
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Create input array (fix other params at midpoint)
    X_grid = np.zeros((n_grid * n_grid, len(param_names)))
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        X_grid[:, i] = (low + high) / 2  # Midpoint
    
    X_grid[:, param_idx1] = P1.flatten()
    X_grid[:, param_idx2] = P2.flatten()
    
    # Predict
    penalties_pred = emulator.predict(X_grid, return_std=False)
    Z = penalties_pred[:, penalty_idx].reshape(n_grid, n_grid)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour
    cf = ax.contourf(P1, P2, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Penalty')
    
    # Training points if provided
    if X_train is not None and y_train is not None:
        sc = ax.scatter(X_train[:, param_idx1], X_train[:, param_idx2],
                       c=y_train[:, penalty_idx], cmap='viridis',
                       edgecolors='white', s=50, linewidths=0.5)
    
    ax.set_xlabel(param_names[param_idx1], fontsize=12)
    ax.set_ylabel(param_names[param_idx2], fontsize=12)
    ax.set_title(f'Penalty Landscape (cf. Figure 3)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()