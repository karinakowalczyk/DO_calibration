"""
Improved GP Emulator for Dansgaard-Oeschger Event Statistics

Improvements implemented:
1. Matern ARD kernel (best performer from diagnostics)
2. Log-transform of small-scale parameters
3. Separate GPs per output (different hyperparameters)
4. Option to use fewer PCA components
5. Increased noise tolerance
6. Better uncertainty quantification

Copy this into your Jupyter notebook after your data loading cells.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# IMPROVEMENT 1: Log-transform small-scale parameters
# =============================================================================

def transform_inputs(X, param_names, log_transform_indices=None):
    """
    Apply log transformation to small-scale parameters.
    
    Parameters
    ----------
    X : array (n_samples, n_params)
        Input parameters
    param_names : list
        Names of parameters
    log_transform_indices : list, optional
        Indices of parameters to log-transform. If None, auto-detects small-scale params.
    
    Returns
    -------
    X_transformed : array
        Transformed inputs
    """
    X_transformed = X.copy()
    
    if log_transform_indices is None:
        # Auto-detect: parameters with max < 0.01 are likely small-scale
        log_transform_indices = []
        for i in range(X.shape[1]):
            if X[:, i].max() < 0.01:
                log_transform_indices.append(i)
                print(f"  Log-transforming {param_names[i]} (max={X[:, i].max():.2e})")
    
    for i in log_transform_indices:
        X_transformed[:, i] = np.log10(X[:, i])
    
    return X_transformed, log_transform_indices


# =============================================================================
# IMPROVEMENT 2: Improved GP Emulator with separate GPs per output
# =============================================================================

class GPEmulatorPDF_Improved:
    """
    Improved GP emulator with:
    - Separate GP per output (allows different hyperparameters)
    - ARD kernel (automatic relevance determination)
    - Input transformation support
    - Better noise handling
    """
    
    def __init__(self, n_pca_components=None, kernel_type='matern_ard', 
                 log_transform_inputs=True, verbose=True):
        """
        Parameters
        ----------
        n_pca_components : int, optional
            Number of PCA components to use. If None, uses all provided.
        kernel_type : str
            'matern_ard', 'matern', 'rbf', or 'custom'
        log_transform_inputs : bool
            Whether to log-transform small-scale parameters
        verbose : bool
            Print fitting progress
        """
        self.n_pca_components = n_pca_components
        self.kernel_type = kernel_type
        self.log_transform_inputs = log_transform_inputs
        self.verbose = verbose
        
        self.scaler_X = StandardScaler()
        self.gp_models = {}
        self.is_fitted = False
        self.pca = None
        self.pdf_xpoints = None
        self.log_transform_indices = None
        self.n_params = None
        self.output_names = None
        
    def _get_kernel(self, n_params, output_idx=0, n_pca=5):
        """Get appropriate kernel based on output type."""
        
        if self.kernel_type == 'matern_ard':
            # ARD kernel with different settings for PCA vs summary stats
            if output_idx < n_pca:
                # PCA components - use Matern 1.5 (rougher)
                kernel = (
                    ConstantKernel(1.0, constant_value_bounds=(0.01, 100.0)) * 
                    Matern(
                        length_scale=[1.0] * n_params,
                        length_scale_bounds=(0.01, 100.0),
                        nu=1.5
                    ) + 
                    WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 0.5))
                )
            else:
                # Summary statistics - use Matern 2.5 (smoother)
                kernel = (
                    ConstantKernel(1.0, constant_value_bounds=(0.01, 100.0)) * 
                    Matern(
                        length_scale=[1.0] * n_params,
                        length_scale_bounds=(0.01, 100.0),
                        nu=2.5
                    ) + 
                    WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
                )
        elif self.kernel_type == 'matern':
            kernel = (
                ConstantKernel(1.0) * 
                Matern(length_scale=1.0, nu=1.5) + 
                WhiteKernel(noise_level=1e-2)
            )
        elif self.kernel_type == 'rbf':
            kernel = (
                ConstantKernel(1.0) * 
                RBF(length_scale=1.0) + 
                WhiteKernel(noise_level=1e-2)
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        return kernel
    
    def fit(self, X, Y, pca_components, param_names=None):
        """
        Fit separate GPs for each output.
        
        Parameters
        ----------
        X : array (n_samples, n_params)
            Input parameters
        Y : array (n_samples, n_other_stats)
            Other summary statistics
        pca_components : array (n_samples, n_pca)
            Pre-computed PCA components
        param_names : list, optional
            Parameter names (for logging)
        """
        self.n_params = X.shape[1]
        
        if param_names is None:
            param_names = [f'param_{i}' for i in range(self.n_params)]
        
        # Log-transform inputs if requested
        if self.log_transform_inputs:
            if self.verbose:
                print("Applying log-transform to small-scale parameters:")
            X_transformed, self.log_transform_indices = transform_inputs(X, param_names)
        else:
            X_transformed = X.copy()
            self.log_transform_indices = []
        
        # Limit PCA components if requested
        if self.n_pca_components is not None:
            pca_components = pca_components[:, :self.n_pca_components]
        
        self._n_pca_actual = pca_components.shape[1]
        self._n_other_stats = Y.shape[1]
        
        # Combine outputs
        Y_full = np.hstack([pca_components, Y])
        n_outputs = Y_full.shape[1]
        
        # Create output names for logging
        self.output_names = [f'PCA_{i+1}' for i in range(self._n_pca_actual)]
        self.output_names += ['mean', 'std', 'avg_stadial_duration', 
                              'avg_waiting_time', 'amplitude', 'n_stadials'][:self._n_other_stats]
        
        # Scale inputs
        X_scaled = self.scaler_X.fit_transform(X_transformed)
        
        # Fit separate GP for each output
        if self.verbose:
            print(f"\nFitting {n_outputs} separate GPs...")
        
        for i in range(n_outputs):
            kernel = self._get_kernel(self.n_params, output_idx=i, n_pca=self._n_pca_actual)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42
            )
            
            gp.fit(X_scaled, Y_full[:, i])
            self.gp_models[i] = gp
            
            if self.verbose:
                # Extract learned noise level
                kernel_params = gp.kernel_.get_params()
                print(f"  {self.output_names[i]}: fitted (log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.2f})")
        
        self.is_fitted = True
        
        if self.verbose:
            print("Done!")
        
    def predict(self, X_new, return_std=True):
        """
        Predict PCA components and summary statistics.
        
        Returns
        -------
        pca_pred : array (n_samples, n_pca)
        Y_pred : array (n_samples, n_other_stats)
        pca_std : array (optional)
        Y_std : array (optional)
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit before predicting")
        
        # Apply same transformation
        X_transformed = X_new.copy()
        for i in self.log_transform_indices:
            X_transformed[:, i] = np.log10(X_new[:, i])
        
        X_scaled = self.scaler_X.transform(X_transformed)
        
        # Predict with each GP
        n_outputs = len(self.gp_models)
        n_samples = X_new.shape[0]
        
        means = np.zeros((n_samples, n_outputs))
        stds = np.zeros((n_samples, n_outputs))
        
        for i in range(n_outputs):
            if return_std:
                mean_i, std_i = self.gp_models[i].predict(X_scaled, return_std=True)
                means[:, i] = mean_i
                stds[:, i] = std_i
            else:
                means[:, i] = self.gp_models[i].predict(X_scaled)
        
        # Split outputs
        pca_pred = means[:, :self._n_pca_actual]
        Y_pred = means[:, self._n_pca_actual:]
        
        if return_std:
            pca_std = stds[:, :self._n_pca_actual]
            Y_std = stds[:, self._n_pca_actual:]
            return pca_pred, Y_pred, pca_std, Y_std
        else:
            return pca_pred, Y_pred
    
    def reconstruct_pdf(self, pca_components, x_grid=None):
        """Reconstruct PDF from PCA components."""
        if self.pca is None:
            raise RuntimeError("PCA model not set")
        
        single_sample = pca_components.ndim == 1
        if single_sample:
            pca_components = pca_components.reshape(1, -1)
        
        # Pad with zeros if using fewer components than PCA was trained with
        if pca_components.shape[1] < self.pca.n_components_:
            padding = np.zeros((pca_components.shape[0], 
                               self.pca.n_components_ - pca_components.shape[1]))
            pca_components = np.hstack([pca_components, padding])
        
        pdf_reconstructed = self.pca.inverse_transform(pca_components)
        pdf_reconstructed = np.maximum(pdf_reconstructed, 0)
        
        if x_grid is None:
            x_grid = self.pdf_xpoints
        
        # Normalize
        for i in range(len(pdf_reconstructed)):
            integral = np.trapezoid(pdf_reconstructed[i], x_grid)
            if integral > 0:
                pdf_reconstructed[i] /= integral
        
        return pdf_reconstructed[0] if single_sample else pdf_reconstructed
    
    def get_learned_length_scales(self):
        """Extract learned length scales from ARD kernels."""
        length_scales = {}
        for i, gp in self.gp_models.items():
            kernel = gp.kernel_
            # Navigate kernel structure to find Matern
            if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k2'):
                matern = kernel.k1.k2
                if hasattr(matern, 'length_scale'):
                    length_scales[self.output_names[i]] = matern.length_scale
        return length_scales


# =============================================================================
# IMPROVEMENT 3: Better cross-validation with detailed metrics
# =============================================================================

def evaluate_emulator(X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
                      param_names, emulator_kwargs=None, cv=5, verbose=True):
    """
    Comprehensive evaluation with cross-validation.
    
    Returns detailed metrics and predictions for analysis.
    """
    if emulator_kwargs is None:
        emulator_kwargs = {}
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Storage
    all_results = {
        'pca_r2': [], 'pca_rmse': [],
        'pdf_rmse': [], 'pdf_max_error': [],
    }
    
    n_pca = pca_components.shape[1]
    for i in range(n_pca):
        all_results[f'pca_{i+1}_r2'] = []
        all_results[f'pca_{i+1}_rmse'] = []
    
    output_names = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                    'amplitude', 'n_stadials']
    for name in output_names[:Y.shape[1]]:
        all_results[f'{name}_r2'] = []
        all_results[f'{name}_rmse'] = []
    
    # Collect all predictions for plotting
    all_pca_true, all_pca_pred = [], []
    all_Y_true, all_Y_pred = [], []
    all_pdf_true, all_pdf_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"  Fold {fold+1}/{cv}...", end=' ')
        
        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pca_train, pca_test = pca_components[train_idx], pca_components[test_idx]
        pdf_test = pdfs[test_idx]
        
        # Fit
        emulator = emulator_class(**emulator_kwargs)
        emulator.pca = pca_model
        emulator.pdf_xpoints = pdf_xpoints
        emulator.fit(X_train, Y_train, pca_train, param_names=param_names)
        
        # Predict
        pca_pred, Y_pred = emulator.predict(X_test, return_std=False)
        pdf_pred = emulator.reconstruct_pdf(pca_pred)
        
        # Metrics
        all_results['pca_r2'].append(r2_score(pca_test, pca_pred, multioutput='uniform_average'))
        all_results['pca_rmse'].append(np.sqrt(mean_squared_error(pca_test, pca_pred)))
        all_results['pdf_rmse'].append(np.sqrt(mean_squared_error(pdf_test, pdf_pred)))
        all_results['pdf_max_error'].append(np.max(np.abs(pdf_test - pdf_pred)))
        
        for i in range(n_pca):
            all_results[f'pca_{i+1}_r2'].append(r2_score(pca_test[:, i], pca_pred[:, i]))
            all_results[f'pca_{i+1}_rmse'].append(np.sqrt(mean_squared_error(pca_test[:, i], pca_pred[:, i])))
        
        for j, name in enumerate(output_names[:Y.shape[1]]):
            all_results[f'{name}_r2'].append(r2_score(Y_test[:, j], Y_pred[:, j]))
            all_results[f'{name}_rmse'].append(np.sqrt(mean_squared_error(Y_test[:, j], Y_pred[:, j])))
        
        # Store predictions
        all_pca_true.append(pca_test)
        all_pca_pred.append(pca_pred)
        all_Y_true.append(Y_test)
        all_Y_pred.append(Y_pred)
        all_pdf_true.append(pdf_test)
        all_pdf_pred.append(pdf_pred)
        
        if verbose:
            print(f"PCA R²={all_results['pca_r2'][-1]:.3f}")
    
    # Aggregate
    summary = {}
    for key, values in all_results.items():
        summary[key] = {'mean': np.mean(values), 'std': np.std(values), 'values': values}
    
    # Combine predictions
    predictions = {
        'pca_true': np.vstack(all_pca_true),
        'pca_pred': np.vstack(all_pca_pred),
        'Y_true': np.vstack(all_Y_true),
        'Y_pred': np.vstack(all_Y_pred),
        'pdf_true': np.vstack(all_pdf_true),
        'pdf_pred': np.vstack(all_pdf_pred),
    }
    
    return summary, predictions


def print_evaluation_summary(summary, n_pca=5):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 70)
    print("EMULATOR EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n--- PCA Components ---")
    print(f"Overall R²: {summary['pca_r2']['mean']:.3f} ± {summary['pca_r2']['std']:.3f}")
    print(f"Overall RMSE: {summary['pca_rmse']['mean']:.4f} ± {summary['pca_rmse']['std']:.4f}")
    
    for i in range(n_pca):
        key_r2 = f'pca_{i+1}_r2'
        key_rmse = f'pca_{i+1}_rmse'
        if key_r2 in summary:
            status = "✓" if summary[key_r2]['mean'] > 0.5 else "⚠️"
            print(f"  PCA {i+1}: R²={summary[key_r2]['mean']:.3f}±{summary[key_r2]['std']:.3f}, "
                  f"RMSE={summary[key_rmse]['mean']:.4f}±{summary[key_rmse]['std']:.4f} {status}")
    
    print("\n--- PDF Reconstruction ---")
    print(f"RMSE: {summary['pdf_rmse']['mean']:.4f} ± {summary['pdf_rmse']['std']:.4f}")
    print(f"Max Error: {summary['pdf_max_error']['mean']:.4f} ± {summary['pdf_max_error']['std']:.4f}")
    
    print("\n--- Summary Statistics ---")
    for name in ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 'amplitude', 'n_stadials']:
        key_r2 = f'{name}_r2'
        key_rmse = f'{name}_rmse'
        if key_r2 in summary:
            status = "✓" if summary[key_r2]['mean'] > 0.5 else "⚠️"
            print(f"  {name}: R²={summary[key_r2]['mean']:.3f}±{summary[key_r2]['std']:.3f}, "
                  f"RMSE={summary[key_rmse]['mean']:.4f}±{summary[key_rmse]['std']:.4f} {status}")


# =============================================================================
# IMPROVEMENT 4: Visualization functions
# =============================================================================

def plot_predicted_vs_actual_improved(predictions, n_pca=5, figsize=(16, 10)):
    """Plot predicted vs actual for all outputs."""
    
    output_names = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                    'amplitude', 'n_stadials']
    
    n_other = predictions['Y_true'].shape[1]
    
    fig, axes = plt.subplots(2, max(n_pca, n_other), figsize=figsize)
    
    # PCA components (top row)
    for i in range(n_pca):
        ax = axes[0, i]
        true = predictions['pca_true'][:, i]
        pred = predictions['pca_pred'][:, i]
        
        ax.scatter(true, pred, alpha=0.5, s=20)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        
        r2 = r2_score(true, pred)
        ax.set_xlabel(f'True PCA {i+1}')
        ax.set_ylabel(f'Predicted')
        ax.set_title(f'PCA {i+1} (R²={r2:.3f})')
    
    # Hide extra subplots in top row
    for i in range(n_pca, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Summary statistics (bottom row)
    for i in range(n_other):
        ax = axes[1, i]
        true = predictions['Y_true'][:, i]
        pred = predictions['Y_pred'][:, i]
        
        ax.scatter(true, pred, alpha=0.5, s=20)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        
        r2 = r2_score(true, pred)
        ax.set_xlabel(f'True {output_names[i]}')
        ax.set_ylabel(f'Predicted')
        ax.set_title(f'{output_names[i]} (R²={r2:.3f})')
    
    # Hide extra subplots in bottom row
    for i in range(n_other, axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.suptitle('Predicted vs Actual (All CV Folds)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_pdf_comparison(predictions, pdf_xpoints, n_examples=6, figsize=(14, 8)):
    """Plot example PDF reconstructions."""
    
    n_samples = predictions['pdf_true'].shape[0]
    indices = np.linspace(0, n_samples-1, n_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for ax, idx in zip(axes, indices):
        ax.plot(pdf_xpoints, predictions['pdf_true'][idx], 'b-', lw=2, label='True')
        ax.plot(pdf_xpoints, predictions['pdf_pred'][idx], 'r--', lw=2, label='Predicted')
        
        rmse = np.sqrt(np.mean((predictions['pdf_true'][idx] - predictions['pdf_pred'][idx])**2))
        ax.set_title(f'Sample {idx} (RMSE={rmse:.4f})')
        ax.set_xlabel('AMOC (Sv)')
        ax.set_ylabel('PDF')
        ax.legend(fontsize=8)
    
    plt.suptitle('PDF Reconstruction Examples', fontsize=14)
    plt.tight_layout()
    return fig


def plot_length_scales(emulator, param_names, figsize=(12, 6)):
    """Visualize learned length scales from ARD kernel."""
    
    length_scales = emulator.get_learned_length_scales()
    
    if not length_scales:
        print("No ARD length scales found (kernel may not be ARD type)")
        return None
    
    # Prepare data
    outputs = list(length_scales.keys())
    n_outputs = len(outputs)
    n_params = len(param_names)
    
    data = np.zeros((n_outputs, n_params))
    for i, output in enumerate(outputs):
        ls = length_scales[output]
        if hasattr(ls, '__len__'):
            data[i, :] = ls
        else:
            data[i, :] = ls
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, aspect='auto', cmap='viridis')
    
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticks(range(n_outputs))
    ax.set_yticklabels(outputs)
    
    ax.set_xlabel('Input Parameter')
    ax.set_ylabel('Output')
    ax.set_title('Learned Length Scales (smaller = more important)')
    
    plt.colorbar(im, ax=ax, label='Length Scale')
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN: Run everything
# =============================================================================

if __name__ == "__main__":
    print("This module should be imported and run in your notebook.")
    print("See usage examples below.")


# =============================================================================
# USAGE EXAMPLES (copy into notebook cells)
# =============================================================================
"""
# ============================================================================
# CELL 1: Setup and fit improved emulator
# ============================================================================

# Your parameter names
param_names = [
    'ocn.diff_dia_min', 
    'ocn.drag_topo_fac', 
    'ocn.slope_max',
    'ocn.diff_iso', 
    'ocn.diff_gm', 
    'ocn.diff_dia_max'
]

# Create improved emulator
emulator_improved = GPEmulatorPDF_Improved(
    n_pca_components=4,        # Use only 4 PCA components (drop the noisy 5th)
    kernel_type='matern_ard',  # Best performing kernel
    log_transform_inputs=True, # Transform small-scale parameters
    verbose=True
)

# Set PCA model and grid
emulator_improved.pca = pca_model
emulator_improved.pdf_xpoints = default_stats['x_grid']

# Fit on all data
emulator_improved.fit(X, other_stats, pca_components, param_names=param_names)


# ============================================================================
# CELL 2: Cross-validation evaluation
# ============================================================================

print("\nRunning 5-fold cross-validation...")
summary, predictions = evaluate_emulator(
    X=X,
    Y=other_stats,
    pca_components=pca_components,
    pdfs=pdf_matrix,
    emulator_class=GPEmulatorPDF_Improved,
    pca_model=pca_model,
    pdf_xpoints=default_stats['x_grid'],
    param_names=param_names,
    emulator_kwargs={
        'n_pca_components': 4,
        'kernel_type': 'matern_ard',
        'log_transform_inputs': True,
        'verbose': False
    },
    cv=5,
    verbose=True
)

# Print summary
print_evaluation_summary(summary, n_pca=4)


# ============================================================================
# CELL 3: Visualizations
# ============================================================================

# Predicted vs Actual
fig1 = plot_predicted_vs_actual_improved(predictions, n_pca=4)
plt.savefig('predicted_vs_actual_improved.png', dpi=150, bbox_inches='tight')
plt.show()

# PDF reconstruction examples
fig2 = plot_pdf_comparison(predictions, default_stats['x_grid'])
plt.savefig('pdf_reconstruction_examples.png', dpi=150, bbox_inches='tight')
plt.show()

# Length scales (shows which parameters are most important)
fig3 = plot_length_scales(emulator_improved, param_names)
if fig3:
    plt.savefig('learned_length_scales.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# CELL 4: Compare old vs improved emulator
# ============================================================================

from sklearn.gaussian_process.kernels import Matern

print("=" * 70)
print("COMPARISON: Original vs Improved Emulator")
print("=" * 70)

# Original emulator settings
print("\n--- Original Emulator (5 PCA, Matern 1.5, no transform) ---")
summary_old, _ = evaluate_emulator(
    X=X, Y=other_stats, pca_components=pca_components, pdfs=pdf_matrix,
    emulator_class=GPEmulatorPDF_Improved,
    pca_model=pca_model,
    pdf_xpoints=default_stats['x_grid'],
    param_names=param_names,
    emulator_kwargs={
        'n_pca_components': 5,
        'kernel_type': 'matern',
        'log_transform_inputs': False,
        'verbose': False
    },
    cv=5, verbose=False
)
print(f"PCA R²: {summary_old['pca_r2']['mean']:.3f} ± {summary_old['pca_r2']['std']:.3f}")
print(f"PDF RMSE: {summary_old['pdf_rmse']['mean']:.4f} ± {summary_old['pdf_rmse']['std']:.4f}")

# Improved emulator
print("\n--- Improved Emulator (4 PCA, Matern ARD, log transform) ---")
summary_new, _ = evaluate_emulator(
    X=X, Y=other_stats, pca_components=pca_components, pdfs=pdf_matrix,
    emulator_class=GPEmulatorPDF_Improved,
    pca_model=pca_model,
    pdf_xpoints=default_stats['x_grid'],
    param_names=param_names,
    emulator_kwargs={
        'n_pca_components': 4,
        'kernel_type': 'matern_ard',
        'log_transform_inputs': True,
        'verbose': False
    },
    cv=5, verbose=False
)
print(f"PCA R²: {summary_new['pca_r2']['mean']:.3f} ± {summary_new['pca_r2']['std']:.3f}")
print(f"PDF RMSE: {summary_new['pdf_rmse']['mean']:.4f} ± {summary_new['pdf_rmse']['std']:.4f}")

# Improvement
print(f"\n--- Improvement ---")
print(f"PCA R² improved by: {summary_new['pca_r2']['mean'] - summary_old['pca_r2']['mean']:.3f}")
print(f"PDF RMSE reduced by: {summary_old['pdf_rmse']['mean'] - summary_new['pdf_rmse']['mean']:.4f}")


# ============================================================================
# CELL 5: Final model for MCMC
# ============================================================================

# Train final model on ALL data
print("\nTraining final model on all data...")

final_emulator = GPEmulatorPDF_Improved(
    n_pca_components=4,
    kernel_type='matern_ard',
    log_transform_inputs=True,
    verbose=True
)
final_emulator.pca = pca_model
final_emulator.pdf_xpoints = default_stats['x_grid']
final_emulator.fit(X, other_stats, pca_components, param_names=param_names)

print("\nFinal emulator ready for MCMC!")

# Example prediction
X_test_example = X[0:1]  # First sample as test
pca_pred, Y_pred, pca_std, Y_std = final_emulator.predict(X_test_example, return_std=True)

print(f"\nExample prediction for first sample:")
print(f"  PCA components: {pca_pred[0]}")
print(f"  PCA uncertainty: {pca_std[0]}")
print(f"  Summary stats: {Y_pred[0]}")
"""