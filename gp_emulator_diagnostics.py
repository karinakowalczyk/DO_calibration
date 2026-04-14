"""
GP Emulator Diagnostics for Dansgaard-Oeschger Event Statistics

This script provides comprehensive diagnostics to understand:
1. Parameter space coverage and distribution
2. Cross-validation stability across different splits
3. Sample-level prediction difficulty
4. Residual analysis and model diagnostics

Usage:
    - Import and call individual diagnostic functions
    - Or run as standalone with your data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. PARAMETER SPACE COVERAGE DIAGNOSTICS
# =============================================================================

def plot_parameter_space_coverage(X, param_names, has_DO_behavior=None, figsize=(16, 12)):
    """
    Visualize the parameter space coverage with pairwise scatter plots.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_params)
        Input parameters
    param_names : list of str
        Names of parameters
    has_DO_behavior : array-like, optional
        Boolean mask for DO vs non-DO runs (before filtering)
    """
    n_params = X.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograms
                ax.hist(X[:, i], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(param_names[i] if i == n_params - 1 else '')
            else:
                # Off-diagonal: scatter plots
                ax.scatter(X[:, j], X[:, i], alpha=0.5, s=20, c='steelblue')
                
            if j == 0:
                ax.set_ylabel(param_names[i])
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            
            if i != n_params - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
    
    plt.suptitle('Parameter Space Coverage (DO-behavior runs)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_parameter_marginals(X, param_names, figsize=(14, 8)):
    """
    Plot marginal distributions with statistics.
    """
    n_params = X.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        data = X[:, i]
        
        # Histogram with KDE
        ax.hist(data, bins=25, density=True, alpha=0.7, color='steelblue', 
                edgecolor='black', label='Data')
        
        # KDE
        kde_x = np.linspace(data.min(), data.max(), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')
        
        # Statistics
        mean, std = data.mean(), data.std()
        ax.axvline(mean, color='orange', linestyle='--', lw=2, label=f'Mean: {mean:.3f}')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nμ={mean:.3f}, σ={std:.3f}')
        ax.legend(fontsize=8)
    
    plt.suptitle('Parameter Marginal Distributions', fontsize=14)
    plt.tight_layout()
    return fig


def check_parameter_gaps(X, param_names, n_bins=10):
    """
    Check for gaps or sparse regions in parameter space.
    
    Returns a report on coverage uniformity.
    """
    report = {}
    n_params = X.shape[1]
    
    print("=" * 60)
    print("PARAMETER SPACE COVERAGE ANALYSIS")
    print("=" * 60)
    
    for i, name in enumerate(param_names):
        data = X[:, i]
        
        # Bin the data
        hist, bin_edges = np.histogram(data, bins=n_bins)
        
        # Check for empty bins
        empty_bins = np.sum(hist == 0)
        min_count = hist.min()
        max_count = hist.max()
        cv = hist.std() / hist.mean()  # Coefficient of variation
        
        report[name] = {
            'empty_bins': empty_bins,
            'min_count': min_count,
            'max_count': max_count,
            'cv': cv,
            'range': (data.min(), data.max())
        }
        
        warning = "⚠️ SPARSE" if empty_bins > 0 or cv > 0.5 else "✓ OK"
        print(f"\n{name} {warning}")
        print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Empty bins: {empty_bins}/{n_bins}")
        print(f"  Counts per bin: {min_count} to {max_count}")
        print(f"  Coverage uniformity (CV): {cv:.3f}")
    
    return report


# =============================================================================
# 2. CROSS-VALIDATION STABILITY ANALYSIS
# =============================================================================

def repeated_cv_stability(X, Y, pca_components, pdfs, emulator_class, pca_model, 
                          pdf_xpoints, emulator_params=None, n_splits=5, n_repeats=10,
                          output_names=None):
    """
    Perform repeated k-fold CV to assess stability of metrics.
    
    Returns detailed statistics on how metrics vary across different splits.
    """
    if emulator_params is None:
        emulator_params = {}
    if output_names is None:
        output_names = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                        'amplitude', 'n_stadials']
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # Storage for all metrics
    all_pca_rmses = []
    all_pdf_rmses = []
    all_output_rmses = {name: [] for name in output_names}
    all_pca_r2s = []
    
    n_pca = pca_components.shape[1]
    pca_component_rmses = {f'pca_{i+1}': [] for i in range(n_pca)}
    pca_component_r2s = {f'pca_{i+1}': [] for i in range(n_pca)}
    
    fold_idx = 0
    for train_idx, test_idx in rkf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pca_train, pca_test = pca_components[train_idx], pca_components[test_idx]
        pdfs_test = pdfs[test_idx]
        
        # Fit emulator
        emulator = emulator_class(**emulator_params)
        emulator.pca = pca_model
        emulator.pdf_xpoints = pdf_xpoints
        emulator.fit(X_train, Y_train, pca_train)
        
        # Predict
        pca_pred, Y_pred = emulator.predict(X_test, return_std=False)
        
        # PCA metrics
        pca_rmse = np.sqrt(mean_squared_error(pca_test, pca_pred))
        all_pca_rmses.append(pca_rmse)
        
        pca_r2 = r2_score(pca_test, pca_pred, multioutput='uniform_average')
        all_pca_r2s.append(pca_r2)
        
        # Individual PCA components
        for i in range(n_pca):
            rmse = np.sqrt(mean_squared_error(pca_test[:, i], pca_pred[:, i]))
            r2 = r2_score(pca_test[:, i], pca_pred[:, i])
            pca_component_rmses[f'pca_{i+1}'].append(rmse)
            pca_component_r2s[f'pca_{i+1}'].append(r2)
        
        # PDF reconstruction
        pdf_pred = emulator.reconstruct_pdf(pca_pred)
        pdf_rmse = np.sqrt(mean_squared_error(pdfs_test.flatten(), pdf_pred.flatten()))
        all_pdf_rmses.append(pdf_rmse)
        
        # Output statistics
        for i, name in enumerate(output_names):
            if i < Y_test.shape[1]:
                rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
                all_output_rmses[name].append(rmse)
        
        fold_idx += 1
    
    # Compile results
    results = {
        'pca': {
            'rmse_mean': np.mean(all_pca_rmses),
            'rmse_std': np.std(all_pca_rmses),
            'rmse_min': np.min(all_pca_rmses),
            'rmse_max': np.max(all_pca_rmses),
            'rmse_range': np.max(all_pca_rmses) - np.min(all_pca_rmses),
            'r2_mean': np.mean(all_pca_r2s),
            'r2_std': np.std(all_pca_r2s),
            'all_rmses': all_pca_rmses,
            'all_r2s': all_pca_r2s
        },
        'pdf': {
            'rmse_mean': np.mean(all_pdf_rmses),
            'rmse_std': np.std(all_pdf_rmses),
            'rmse_range': np.max(all_pdf_rmses) - np.min(all_pdf_rmses),
            'all_rmses': all_pdf_rmses
        }
    }
    
    # Add individual PCA components
    for pca_name in pca_component_rmses.keys():
        results[pca_name] = {
            'rmse_mean': np.mean(pca_component_rmses[pca_name]),
            'rmse_std': np.std(pca_component_rmses[pca_name]),
            'r2_mean': np.mean(pca_component_r2s[pca_name]),
            'r2_std': np.std(pca_component_r2s[pca_name]),
            'all_rmses': pca_component_rmses[pca_name],
            'all_r2s': pca_component_r2s[pca_name]
        }
    
    # Add output statistics
    for name in output_names:
        if len(all_output_rmses[name]) > 0:
            results[name] = {
                'rmse_mean': np.mean(all_output_rmses[name]),
                'rmse_std': np.std(all_output_rmses[name]),
                'rmse_range': np.max(all_output_rmses[name]) - np.min(all_output_rmses[name]),
                'all_rmses': all_output_rmses[name]
            }
    
    return results


def plot_cv_stability(cv_results, figsize=(14, 10)):
    """
    Plot the stability of cross-validation metrics across different splits.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. PCA RMSE distribution
    ax = axes[0, 0]
    ax.hist(cv_results['pca']['all_rmses'], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(cv_results['pca']['rmse_mean'], color='red', linestyle='--', 
               label=f"Mean: {cv_results['pca']['rmse_mean']:.4f}")
    ax.set_xlabel('PCA RMSE')
    ax.set_ylabel('Count')
    ax.set_title('PCA RMSE Distribution Across CV Folds')
    ax.legend()
    
    # 2. PDF RMSE distribution
    ax = axes[0, 1]
    ax.hist(cv_results['pdf']['all_rmses'], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(cv_results['pdf']['rmse_mean'], color='red', linestyle='--',
               label=f"Mean: {cv_results['pdf']['rmse_mean']:.4f}")
    ax.set_xlabel('PDF RMSE')
    ax.set_ylabel('Count')
    ax.set_title('PDF RMSE Distribution Across CV Folds')
    ax.legend()
    
    # 3. Individual PCA component R² 
    ax = axes[1, 0]
    pca_keys = [k for k in cv_results.keys() if k.startswith('pca_')]
    r2_means = [cv_results[k]['r2_mean'] for k in pca_keys]
    r2_stds = [cv_results[k]['r2_std'] for k in pca_keys]
    
    x_pos = np.arange(len(pca_keys))
    ax.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pca_keys)
    ax.set_ylabel('R²')
    ax.set_title('R² per PCA Component (mean ± std)')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.legend()
    
    # 4. Output statistics RMSE variability
    ax = axes[1, 1]
    output_keys = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                   'amplitude', 'n_stadials']
    output_keys = [k for k in output_keys if k in cv_results]
    
    # Normalize by mean to show relative variability
    rel_stds = []
    labels = []
    for k in output_keys:
        if cv_results[k]['rmse_mean'] > 0:
            rel_std = cv_results[k]['rmse_std'] / cv_results[k]['rmse_mean']
            rel_stds.append(rel_std)
            labels.append(k)
    
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, rel_stds, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Relative Std (σ/μ)')
    ax.set_title('RMSE Variability Across CV Folds')
    
    plt.tight_layout()
    return fig


def print_cv_stability_report(cv_results, n_splits=5, n_repeats=10):
    """
    Print a detailed report on CV stability.
    """
    print("=" * 70)
    print(f"CROSS-VALIDATION STABILITY REPORT ({n_splits}-fold × {n_repeats} repeats)")
    print("=" * 70)
    
    print("\n--- PCA Components (Overall) ---")
    print(f"RMSE: {cv_results['pca']['rmse_mean']:.4f} ± {cv_results['pca']['rmse_std']:.4f}")
    print(f"  Range: [{cv_results['pca']['rmse_min']:.4f}, {cv_results['pca']['rmse_max']:.4f}]")
    print(f"  Variability (range/mean): {cv_results['pca']['rmse_range']/cv_results['pca']['rmse_mean']:.2%}")
    print(f"R²: {cv_results['pca']['r2_mean']:.4f} ± {cv_results['pca']['r2_std']:.4f}")
    
    print("\n--- Individual PCA Components ---")
    for i in range(1, 10):
        key = f'pca_{i}'
        if key not in cv_results:
            break
        r = cv_results[key]
        stability = "⚠️ UNSTABLE" if r['r2_std'] > 0.1 or r['r2_mean'] < 0.5 else "✓ STABLE"
        print(f"  PCA {i}: R²={r['r2_mean']:.3f}±{r['r2_std']:.3f}, "
              f"RMSE={r['rmse_mean']:.4f}±{r['rmse_std']:.4f} {stability}")
    
    print("\n--- PDF Reconstruction ---")
    print(f"RMSE: {cv_results['pdf']['rmse_mean']:.4f} ± {cv_results['pdf']['rmse_std']:.4f}")
    print(f"  Variability: {cv_results['pdf']['rmse_std']/cv_results['pdf']['rmse_mean']:.2%}")
    
    print("\n--- Other Summary Statistics ---")
    for name in ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                 'amplitude', 'n_stadials']:
        if name in cv_results:
            r = cv_results[name]
            variability = r['rmse_std'] / r['rmse_mean'] if r['rmse_mean'] > 0 else 0
            stability = "⚠️ HIGH VAR" if variability > 0.3 else "✓ OK"
            print(f"  {name}: RMSE={r['rmse_mean']:.4f}±{r['rmse_std']:.4f} "
                  f"(var: {variability:.1%}) {stability}")


# =============================================================================
# 3. SAMPLE-LEVEL PREDICTION DIFFICULTY
# =============================================================================

def identify_hard_samples(X, Y, pca_components, pdfs, emulator_class, pca_model, 
                          pdf_xpoints, emulator_params=None, n_splits=5, n_repeats=5):
    """
    Identify which samples are consistently hard to predict across CV folds.
    
    Returns indices and error statistics for each sample.
    """
    if emulator_params is None:
        emulator_params = {}
    
    n_samples = X.shape[0]
    
    # Track errors for each sample
    sample_pca_errors = {i: [] for i in range(n_samples)}
    sample_pdf_errors = {i: [] for i in range(n_samples)}
    sample_test_counts = {i: 0 for i in range(n_samples)}
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pca_train, pca_test = pca_components[train_idx], pca_components[test_idx]
        pdfs_test = pdfs[test_idx]
        
        emulator = emulator_class(**emulator_params)
        emulator.pca = pca_model
        emulator.pdf_xpoints = pdf_xpoints
        emulator.fit(X_train, Y_train, pca_train)
        
        pca_pred, _ = emulator.predict(X_test, return_std=False)
        pdf_pred = emulator.reconstruct_pdf(pca_pred)
        
        # Track per-sample errors
        for local_idx, global_idx in enumerate(test_idx):
            pca_error = np.sqrt(np.mean((pca_test[local_idx] - pca_pred[local_idx])**2))
            pdf_error = np.sqrt(np.mean((pdfs_test[local_idx] - pdf_pred[local_idx])**2))
            
            sample_pca_errors[global_idx].append(pca_error)
            sample_pdf_errors[global_idx].append(pdf_error)
            sample_test_counts[global_idx] += 1
    
    # Compute statistics
    sample_stats = []
    for i in range(n_samples):
        if sample_test_counts[i] > 0:
            pca_errs = sample_pca_errors[i]
            pdf_errs = sample_pdf_errors[i]
            sample_stats.append({
                'index': i,
                'pca_error_mean': np.mean(pca_errs),
                'pca_error_std': np.std(pca_errs),
                'pdf_error_mean': np.mean(pdf_errs),
                'pdf_error_std': np.std(pdf_errs),
                'n_tests': sample_test_counts[i]
            })
    
    return sample_stats


def plot_hard_samples(X, sample_stats, param_names, n_hardest=20, figsize=(14, 10)):
    """
    Visualize which samples are hardest to predict and their location in parameter space.
    """
    # Sort by mean PCA error
    sample_stats_sorted = sorted(sample_stats, key=lambda x: x['pca_error_mean'], reverse=True)
    hard_indices = [s['index'] for s in sample_stats_sorted[:n_hardest]]
    easy_indices = [s['index'] for s in sample_stats_sorted[-n_hardest:]]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot each parameter dimension
    for ax_idx, (ax, name) in enumerate(zip(axes.flatten(), param_names)):
        # All samples
        all_errors = [s['pca_error_mean'] for s in sample_stats]
        all_x = X[:, ax_idx]
        
        scatter = ax.scatter(all_x, all_errors, alpha=0.5, c='gray', s=30, label='All')
        
        # Highlight hard samples
        hard_x = X[hard_indices, ax_idx]
        hard_errors = [sample_stats_sorted[i]['pca_error_mean'] for i in range(n_hardest)]
        ax.scatter(hard_x, hard_errors, c='red', s=50, marker='x', label='Hardest', zorder=5)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Mean PCA Error')
        ax.legend(fontsize=8)
    
    plt.suptitle('Sample Prediction Difficulty vs Parameter Values\n(Red X = hardest samples)', 
                 fontsize=12)
    plt.tight_layout()
    return fig, hard_indices


def plot_error_distribution(sample_stats, figsize=(12, 5)):
    """
    Plot the distribution of per-sample errors.
    """
    pca_errors = [s['pca_error_mean'] for s in sample_stats]
    pdf_errors = [s['pdf_error_mean'] for s in sample_stats]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PCA error distribution
    ax = axes[0]
    ax.hist(pca_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(pca_errors), color='red', linestyle='--', 
               label=f'Median: {np.median(pca_errors):.4f}')
    ax.axvline(np.percentile(pca_errors, 90), color='orange', linestyle='--',
               label=f'90th %ile: {np.percentile(pca_errors, 90):.4f}')
    ax.set_xlabel('Mean PCA Error (per sample)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-Sample PCA Errors')
    ax.legend()
    
    # PDF error distribution
    ax = axes[1]
    ax.hist(pdf_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(pdf_errors), color='red', linestyle='--',
               label=f'Median: {np.median(pdf_errors):.4f}')
    ax.axvline(np.percentile(pdf_errors, 90), color='orange', linestyle='--',
               label=f'90th %ile: {np.percentile(pdf_errors, 90):.4f}')
    ax.set_xlabel('Mean PDF Error (per sample)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-Sample PDF Errors')
    ax.legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# 4. RESIDUAL ANALYSIS
# =============================================================================

def residual_analysis(X, Y, pca_components, pdfs, emulator_class, pca_model, 
                      pdf_xpoints, param_names, emulator_params=None, test_size=0.2):
    """
    Analyze residuals for model misspecification.
    
    Checks:
    - Residual vs predicted (should be random scatter)
    - Residual vs input parameters (should be random)
    - Residual normality
    - Residual autocorrelation
    """
    from sklearn.model_selection import train_test_split
    
    if emulator_params is None:
        emulator_params = {}
    
    # Single train/test split for detailed analysis
    X_train, X_test, pca_train, pca_test, Y_train, Y_test, pdf_train, pdf_test = \
        train_test_split(X, pca_components, Y, pdfs, test_size=test_size, random_state=42)
    
    emulator = emulator_class(**emulator_params)
    emulator.pca = pca_model
    emulator.pdf_xpoints = pdf_xpoints
    emulator.fit(X_train, Y_train, pca_train)
    
    pca_pred, Y_pred = emulator.predict(X_test, return_std=False)
    
    # Compute residuals for first PCA component
    residuals = pca_test - pca_pred
    
    results = {
        'X_test': X_test,
        'pca_test': pca_test,
        'pca_pred': pca_pred,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'residuals': residuals
    }
    
    return results


def plot_residual_diagnostics(residual_results, param_names, figsize=(16, 12)):
    """
    Plot comprehensive residual diagnostics.
    """
    X_test = residual_results['X_test']
    pca_pred = residual_results['pca_pred']
    residuals = residual_results['residuals']
    
    n_pca = residuals.shape[1]
    
    fig = plt.figure(figsize=figsize)
    
    # 1. Residuals vs Predicted for each PCA component
    for i in range(min(n_pca, 3)):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.scatter(pca_pred[:, i], residuals[:, i], alpha=0.6, s=30)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel(f'Predicted PCA {i+1}')
        ax.set_ylabel(f'Residual PCA {i+1}')
        ax.set_title(f'Residuals vs Predicted (PCA {i+1})')
    
    # 2. Residuals vs Input Parameters for PCA 1
    for i, name in enumerate(param_names[:3]):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        ax.scatter(X_test[:, i], residuals[:, 0], alpha=0.6, s=30)
        ax.axhline(0, color='red', linestyle='--')
        
        # Compute correlation
        corr = np.corrcoef(X_test[:, i], residuals[:, 0])[0, 1]
        ax.set_xlabel(name)
        ax.set_ylabel('Residual PCA 1')
        ax.set_title(f'Residual vs {name}\n(r={corr:.3f})')
    
    # 3. Q-Q plots for normality
    for i in range(min(n_pca, 3)):
        ax = fig.add_subplot(3, 4, 8 + i + 1)
        stats.probplot(residuals[:, i], dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (PCA {i+1})')
    
    # 4. Histogram of residuals for PCA 1
    ax = fig.add_subplot(3, 4, 12)
    ax.hist(residuals[:, 0], bins=20, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay normal distribution
    mu, std = residuals[:, 0].mean(), residuals[:, 0].std()
    x = np.linspace(residuals[:, 0].min(), residuals[:, 0].max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label='Normal')
    ax.set_xlabel('Residual PCA 1')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    
    plt.suptitle('Residual Diagnostics', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compute_residual_statistics(residual_results, param_names):
    """
    Compute and print residual statistics.
    """
    residuals = residual_results['residuals']
    X_test = residual_results['X_test']
    
    print("=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)
    
    n_pca = residuals.shape[1]
    
    print("\n--- Basic Statistics ---")
    for i in range(n_pca):
        r = residuals[:, i]
        print(f"PCA {i+1}: mean={r.mean():.4f}, std={r.std():.4f}, "
              f"skew={stats.skew(r):.3f}, kurtosis={stats.kurtosis(r):.3f}")
    
    print("\n--- Normality Tests (Shapiro-Wilk) ---")
    for i in range(n_pca):
        stat, p_value = stats.shapiro(residuals[:, i])
        status = "✓ Normal" if p_value > 0.05 else "⚠️ Non-normal"
        print(f"PCA {i+1}: p-value={p_value:.4f} {status}")
    
    print("\n--- Correlation with Input Parameters ---")
    print("(Values > 0.2 suggest model misspecification)")
    
    for i in range(n_pca):
        correlations = []
        for j, name in enumerate(param_names):
            corr = np.corrcoef(X_test[:, j], residuals[:, i])[0, 1]
            correlations.append((name, corr))
        
        print(f"\nPCA {i+1}:")
        for name, corr in correlations:
            warning = "⚠️" if abs(corr) > 0.2 else ""
            print(f"  {name}: r={corr:.3f} {warning}")


# =============================================================================
# 5. PREDICTED VS ACTUAL PLOTS
# =============================================================================

def plot_predicted_vs_actual(X, Y, pca_components, pdfs, emulator_class, pca_model,
                             pdf_xpoints, emulator_params=None, n_splits=5, figsize=(16, 12)):
    """
    Create predicted vs actual plots for all outputs across CV folds.
    """
    if emulator_params is None:
        emulator_params = {}
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Collect all predictions
    all_pca_true = []
    all_pca_pred = []
    all_Y_true = []
    all_Y_pred = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        pca_train, pca_test = pca_components[train_idx], pca_components[test_idx]
        
        emulator = emulator_class(**emulator_params)
        emulator.pca = pca_model
        emulator.pdf_xpoints = pdf_xpoints
        emulator.fit(X_train, Y_train, pca_train)
        
        pca_pred, Y_pred = emulator.predict(X_test, return_std=False)
        
        all_pca_true.append(pca_test)
        all_pca_pred.append(pca_pred)
        all_Y_true.append(Y_test)
        all_Y_pred.append(Y_pred)
    
    all_pca_true = np.vstack(all_pca_true)
    all_pca_pred = np.vstack(all_pca_pred)
    all_Y_true = np.vstack(all_Y_true)
    all_Y_pred = np.vstack(all_Y_pred)
    
    n_pca = all_pca_true.shape[1]
    output_names = ['mean', 'std', 'avg_stadial_duration', 'avg_waiting_time', 
                    'amplitude', 'n_stadials']
    
    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=figsize)
    
    # Top row: PCA components
    for i in range(min(n_pca, 5)):
        ax = axes[0, i]
        true = all_pca_true[:, i]
        pred = all_pca_pred[:, i]
        
        ax.scatter(true, pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        
        r2 = r2_score(true, pred)
        ax.set_xlabel(f'True PCA {i+1}')
        ax.set_ylabel(f'Predicted PCA {i+1}')
        ax.set_title(f'PCA {i+1}\nR²={r2:.3f}')
        ax.legend(fontsize=8)
    
    # Hide extra subplot in top row
    axes[0, 5].axis('off')
    
    # Bottom row: Output statistics
    for i, name in enumerate(output_names):
        ax = axes[1, i]
        true = all_Y_true[:, i]
        pred = all_Y_pred[:, i]
        
        ax.scatter(true, pred, alpha=0.5, s=20)
        
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        
        r2 = r2_score(true, pred)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted')
        ax.set_title(f'{name}\nR²={r2:.3f}')
    
    plt.suptitle('Predicted vs Actual (All CV Folds Combined)', fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# 6. MAIN DIAGNOSTIC RUNNER
# =============================================================================

def run_all_diagnostics(X, Y, pca_components, pdfs, pca_model, pdf_xpoints,
                        param_names, emulator_class, emulator_params=None,
                        output_dir='.', n_cv_repeats=10):
    """
    Run all diagnostics and save figures.
    
    Parameters
    ----------
    X : array (n_samples, n_params)
        Input parameters
    Y : array (n_samples, n_outputs)
        Other summary statistics
    pca_components : array (n_samples, n_pca)
        Pre-computed PCA components
    pdfs : array (n_samples, n_grid)
        PDF values
    pca_model : fitted PCA
        The PCA model for reconstruction
    pdf_xpoints : array
        Grid points for PDF
    param_names : list
        Names of input parameters
    emulator_class : class
        The GP emulator class
    emulator_params : dict, optional
        Parameters for the emulator
    output_dir : str
        Directory to save figures
    n_cv_repeats : int
        Number of CV repeats for stability analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if emulator_params is None:
        emulator_params = {}
    
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE GP EMULATOR DIAGNOSTICS")
    print("=" * 70)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} parameters")
    print(f"PCA components: {pca_components.shape[1]}")
    print(f"Other outputs: {Y.shape[1]}")
    
    # 1. Parameter Space Coverage
    print("\n" + "-" * 50)
    print("1. PARAMETER SPACE COVERAGE")
    print("-" * 50)
    
    fig1 = plot_parameter_space_coverage(X, param_names)
    fig1.savefig(f'{output_dir}/01_parameter_space_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plot_parameter_marginals(X, param_names)
    fig2.savefig(f'{output_dir}/02_parameter_marginals.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    gap_report = check_parameter_gaps(X, param_names)
    
    # 2. Cross-Validation Stability
    print("\n" + "-" * 50)
    print("2. CROSS-VALIDATION STABILITY")
    print("-" * 50)
    
    cv_results = repeated_cv_stability(
        X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
        emulator_params=emulator_params, n_splits=5, n_repeats=n_cv_repeats
    )
    
    print_cv_stability_report(cv_results, n_splits=5, n_repeats=n_cv_repeats)
    
    fig3 = plot_cv_stability(cv_results)
    fig3.savefig(f'{output_dir}/03_cv_stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 3. Sample-Level Difficulty
    print("\n" + "-" * 50)
    print("3. SAMPLE-LEVEL PREDICTION DIFFICULTY")
    print("-" * 50)
    
    sample_stats = identify_hard_samples(
        X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
        emulator_params=emulator_params, n_splits=5, n_repeats=5
    )
    
    fig4, hard_indices = plot_hard_samples(X, sample_stats, param_names)
    fig4.savefig(f'{output_dir}/04_hard_samples_vs_params.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    fig5 = plot_error_distribution(sample_stats)
    fig5.savefig(f'{output_dir}/05_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    print(f"\nHardest {len(hard_indices)} samples (indices): {hard_indices}")
    
    # 4. Residual Analysis
    print("\n" + "-" * 50)
    print("4. RESIDUAL ANALYSIS")
    print("-" * 50)
    
    residual_results = residual_analysis(
        X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
        param_names, emulator_params=emulator_params
    )
    
    compute_residual_statistics(residual_results, param_names)
    
    fig6 = plot_residual_diagnostics(residual_results, param_names)
    fig6.savefig(f'{output_dir}/06_residual_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    
    # 5. Predicted vs Actual
    print("\n" + "-" * 50)
    print("5. PREDICTED VS ACTUAL")
    print("-" * 50)
    
    fig7 = plot_predicted_vs_actual(
        X, Y, pca_components, pdfs, emulator_class, pca_model, pdf_xpoints,
        emulator_params=emulator_params
    )
    fig7.savefig(f'{output_dir}/07_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close(fig7)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {output_dir}/")
    print("  01_parameter_space_pairplot.png")
    print("  02_parameter_marginals.png")
    print("  03_cv_stability.png")
    print("  04_hard_samples_vs_params.png")
    print("  05_error_distribution.png")
    print("  06_residual_diagnostics.png")
    print("  07_predicted_vs_actual.png")
    
    return {
        'cv_results': cv_results,
        'sample_stats': sample_stats,
        'hard_indices': hard_indices,
        'residual_results': residual_results,
        'gap_report': gap_report
    }


# =============================================================================
# EXAMPLE USAGE (uncomment and modify for your data)
# =============================================================================
"""
# After running your existing code to get X, other_stats, pca_components, pdf_matrix, etc.

from gp_emulator_diagnostics import run_all_diagnostics
from gp_emulator_pdf import GPEmulatorPDF  # Your emulator class

param_names = [
    'ocn.diff_dia_min', 'ocn.drag_topo_fac', 'ocn.slope_max',
    'ocn.diff_iso', 'ocn.diff_gm', 'ocn.diff_dia_max'
]

kernel = 1 * Matern(length_scale=1.0, nu=1.5)
emulator_params = {'kernel': kernel}

results = run_all_diagnostics(
    X=X,
    Y=other_stats,
    pca_components=pca_components,
    pdfs=pdf_matrix,
    pca_model=pca_model,
    pdf_xpoints=default_stats['x_grid'],
    param_names=param_names,
    emulator_class=GPEmulatorPDF,
    emulator_params=emulator_params,
    output_dir='./diagnostics',
    n_cv_repeats=10
)

# Access specific results:
cv_stability = results['cv_results']
hard_samples = results['hard_indices']
"""