import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class GPEmulatorPDF:
    """
    GP emulator mapping ocean parameters → PCA components of the AMOC PDF + scalar statistics.

    A single multi-output GP is fitted on the combined output vector
    [PCA_1, ..., PCA_n, mean, std, avg_stadial_duration, avg_waiting_time, avg_amplitude, n_stadials].
    PCA components can be inverted via the stored pca model to reconstruct full AMOC PDFs.

    Attributes set externally before calling fit/predict:
        pca         : fitted sklearn PCA model (from PCA.fit on ensemble PDFs)
        pdf_xpoints : 1-D array, the fixed AMOC x-grid used when evaluating KDEs
    """

    def __init__(self, kernel=None, scaler_X=None, scaler_y=None):
        self.kernel = kernel if kernel is not None else RBF(1.0) + WhiteKernel(1e-6)
        self.scaler_X = scaler_X if scaler_X is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()
        self.pca = None
        self.gp_model = None
        self.is_fitted = False
        self.pdf_xpoints = None
        self.n_pca_components = None

    def fit(self, X, Y, pca_components):
        """
        Parameters
        ----------
        X : (n_samples, n_params)  — ocean parameter values
        Y : (n_samples, n_stats)   — scalar stats [mean, std, avg_stadial_duration,
                                     avg_waiting_time, avg_amplitude, n_stadials]
        pca_components : (n_samples, n_pca) — pre-computed PCA scores of the AMOC PDF
        """
        self.n_pca_components = pca_components.shape[1]
        Y_full = np.hstack([pca_components, Y])

        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y_full)

        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
        )
        self.gp_model.fit(X_scaled, Y_scaled)
        self.is_fitted = True

    def predict(self, X_new, return_std=True):
        """
        Returns
        -------
        pca_pred  : (n, n_pca)   predicted PCA scores
        Y_pred    : (n, n_stats) predicted scalar statistics
        pca_std   : (n, n_pca)   predictive std  [only if return_std=True]
        Y_std     : (n, n_stats) predictive std  [only if return_std=True]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X_scaled = self.scaler_X.transform(X_new)

        if return_std:
            mean_scaled, std_scaled = self.gp_model.predict(X_scaled, return_std=True)
        else:
            mean_scaled = self.gp_model.predict(X_scaled)
            std_scaled = None

        mean_full = self.scaler_y.inverse_transform(mean_scaled)

        pca_pred = mean_full[:, :self.n_pca_components]
        Y_pred = mean_full[:, self.n_pca_components:]

        if return_std and std_scaled is not None:
            std_full = std_scaled * self.scaler_y.scale_
            pca_std = std_full[:, :self.n_pca_components]
            Y_std = std_full[:, self.n_pca_components:]
            return pca_pred, Y_pred, pca_std, Y_std

        return pca_pred, Y_pred

    def reconstruct_pdf(self, pca_components, x_grid=None):
        """
        Inverse-PCA transform PCA scores back to a normalised PDF on x_grid.

        Parameters
        ----------
        pca_components : (n, n_pca) or (n_pca,)
        x_grid : array, optional — defaults to self.pdf_xpoints

        Returns
        -------
        pdf : (n, n_grid) or (n_grid,) — non-negative, unit-integral PDFs
        """
        if self.pca is None:
            raise RuntimeError("Set emulator.pca before calling reconstruct_pdf().")

        single = pca_components.ndim == 1
        if single:
            pca_components = pca_components.reshape(1, -1)

        pdf = self.pca.inverse_transform(pca_components)
        pdf = np.maximum(pdf, 0)

        if x_grid is None:
            if self.pdf_xpoints is None:
                raise ValueError("Provide x_grid or set emulator.pdf_xpoints.")
            x_grid = self.pdf_xpoints

        for i in range(len(pdf)):
            integral = np.trapezoid(pdf[i], x_grid)
            if integral > 0:
                pdf[i] /= integral

        return pdf[0] if single else pdf


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

STAT_NAMES = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time',
              'avg_amplitude', 'n_stadials']


def crossvalidation(X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
                    emulator_kwargs=None, cv=5, output_names=None):
    """
    K-fold cross-validation for a GPEmulatorPDF-like class.

    Parameters
    ----------
    X              : (n, n_params)
    Y              : (n, n_stats)  scalar statistics
    pca_components : (n, n_pca)
    pdfs           : (n, n_grid)   true KDE PDFs on pdf_xpoints
    emulator_class : class to instantiate (must accept **emulator_kwargs)
    pca_model      : fitted sklearn PCA model
    pdf_xpoints    : 1-D array, fixed AMOC x-grid
    emulator_kwargs: dict, passed to emulator_class constructor
    cv             : int, number of folds
    output_names   : list of stat names matching columns of Y

    Returns
    -------
    results : dict with keys 'pca', 'pdf', 'pca_1'..'pca_n', and one key per
              stat name.  Each value has 'mean', 'std', 'scores'.
    """
    if emulator_kwargs is None:
        emulator_kwargs = {}
    if output_names is None:
        output_names = STAT_NAMES[:Y.shape[1]]

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    n_pca = pca_components.shape[1]

    pca_rmses, pdf_rmses, pdf_sup_diffs = [], [], []
    pca_comp_rmses = {f'pca_{i+1}': [] for i in range(n_pca)}
    stat_rmses = {name: [] for name in output_names}

    for train_idx, test_idx in kf.split(X):
        em = emulator_class(**emulator_kwargs)
        em.pca = pca_model
        em.pdf_xpoints = pdf_xpoints
        em.fit(X[train_idx], Y[train_idx], pca_components[train_idx])

        pca_pred, Y_pred = em.predict(X[test_idx], return_std=False)
        pdf_pred = em.reconstruct_pdf(pca_pred)

        pca_rmses.append(np.sqrt(mean_squared_error(pca_components[test_idx], pca_pred)))
        pdf_rmses.append(np.sqrt(mean_squared_error(pdfs[test_idx], pdf_pred)))
        pdf_sup_diffs.append(np.max(np.abs(pdfs[test_idx] - pdf_pred)))

        for i in range(n_pca):
            pca_comp_rmses[f'pca_{i+1}'].append(
                np.sqrt(mean_squared_error(pca_components[test_idx, i], pca_pred[:, i]))
            )
        for j, name in enumerate(output_names):
            if j < Y.shape[1]:
                stat_rmses[name].append(
                    np.sqrt(mean_squared_error(Y[test_idx, j], Y_pred[:, j]))
                )

    def _agg(scores):
        return {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}

    results = {
        'pca': {'rmse_mean': np.mean(pca_rmses), 'rmse_std': np.std(pca_rmses),
                'rmse_scores': pca_rmses},
        'pdf': {'rmse_mean': np.mean(pdf_rmses), 'rmse_std': np.std(pdf_rmses),
                'rmse_scores': pdf_rmses,
                'sup_diff_mean': np.mean(pdf_sup_diffs), 'sup_diff_std': np.std(pdf_sup_diffs),
                'sup_diff_scores': pdf_sup_diffs},
    }
    for key, scores in pca_comp_rmses.items():
        results[key] = _agg(scores)
    for name, scores in stat_rmses.items():
        results[name] = _agg(scores)

    return results


def print_cv_summary(cv_results, output_names=None):
    """Print a readable cross-validation summary."""
    if output_names is None:
        output_names = STAT_NAMES

    print("=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print("\n--- PCA Components ---")
    r = cv_results['pca']
    print(f"  Overall RMSE : {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
    i = 1
    while f'pca_{i}' in cv_results:
        r = cv_results[f'pca_{i}']
        print(f"  PCA {i}       : {r['mean']:.4f} ± {r['std']:.4f}")
        i += 1

    print("\n--- PDF Reconstruction ---")
    r = cv_results['pdf']
    print(f"  RMSE         : {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
    print(f"  Sup-norm     : {r['sup_diff_mean']:.4f} ± {r['sup_diff_std']:.4f}")

    print("\n--- Scalar Statistics ---")
    for name in output_names:
        if name in cv_results:
            r = cv_results[name]
            print(f"  {name:<25}: {r['mean']:.4f} ± {r['std']:.4f}")
    print("=" * 60)
