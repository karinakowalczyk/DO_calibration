# # 2. Helper Functions
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def lowpass_filter(data, cutoff_years, dt=1.0):
    """Apply a lowpass Butterworth filter."""
    cutoff_freq = 1.0 / cutoff_years
    nyquist_freq = 1.0 / (2.0 * dt)
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def plot_correlation_heatmap(X, y, param_names, output_names, title, figsize=(14, 10)):
    """Heat map of parameter-output correlations (Figures 2a, 9a/b)."""
    n_params = X.shape[1]
    n_outputs = y.shape[1]
    
    correlations = np.zeros((n_outputs, n_params))
    for i in range(n_outputs):
        for j in range(n_params):
            corr = np.corrcoef(y[:, i], X[:, j])[0, 1]
            correlations[i, j] = corr if np.isfinite(corr) else 0
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=12)
    
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
                ax.plot(j, i, 'ko', markersize=6)
    
    plt.tight_layout()
    plt.show()
    return correlations


def plot_emulator_validation(y_true, y_pred, output_names, figsize=(16, 12)):
    """Emulator validation plots (Figure 4)."""
    n_outputs = y_true.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_outputs):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=30, c='steelblue')
        
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
                'r--', linewidth=2)
        
        r2 = r2_score(true_vals, pred_vals)
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{output_names[i]}\nR² = {r2:.3f}', fontsize=11)
        ax.grid(alpha=0.3)
    
    for i in range(n_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('NN Ensemble Emulator vs Actual Model (cf. Figure 4)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_penalty_landscape_2d(emulator, X_train, y_train, param_names, param_bounds,
                               idx1, idx2, penalty_idx, penalty_name, n_grid=50):
    """Plot emulated penalty landscape (Figure 3)."""
    p1_min, p1_max = param_bounds[param_names[idx1]]
    p2_min, p2_max = param_bounds[param_names[idx2]]
    
    p1_range = np.linspace(p1_min, p1_max, n_grid)
    p2_range = np.linspace(p2_min, p2_max, n_grid)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    X_grid = np.tile(np.median(X_train, axis=0), (n_grid * n_grid, 1))
    X_grid[:, idx1] = P1.flatten()
    X_grid[:, idx2] = P2.flatten()
    
    penalties_pred = emulator.predict(X_grid, return_std=False)
    Z = penalties_pred[:, penalty_idx].reshape(n_grid, n_grid)
    
    # Normalize for display
    P1_norm = (P1 - p1_min) / (p1_max - p1_min)
    P2_norm = (P2 - p2_min) / (p2_max - p2_min)
    X1_norm = (X_train[:, idx1] - p1_min) / (p1_max - p1_min)
    X2_norm = (X_train[:, idx2] - p2_min) / (p2_max - p2_min)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (a) Training points
    ax = axes[0]
    sc = ax.scatter(X1_norm, X2_norm, c=y_train[:, penalty_idx],
                   cmap='viridis', s=50, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Penalty')
    ax.set_xlabel(f'{param_names[idx1]} (norm)', fontsize=11)
    ax.set_ylabel(f'{param_names[idx2]} (norm)', fontsize=11)
    ax.set_title('(a) Training Points', fontsize=12)
    
    # (b) NN surface with points
    ax = axes[1]
    cf = ax.contourf(P1_norm, P2_norm, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Penalty')
    ax.scatter(X1_norm, X2_norm, c='white', s=20, edgecolors='black', linewidths=0.5, alpha=0.7)
    ax.set_xlabel(f'{param_names[idx1]} (norm)', fontsize=11)
    ax.set_ylabel(f'{param_names[idx2]} (norm)', fontsize=11)
    ax.set_title('(b) NN Surface + Points', fontsize=12)
    
    # (c) NN surface only
    ax = axes[2]
    cf = ax.contourf(P1_norm, P2_norm, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Penalty')
    ax.set_xlabel(f'{param_names[idx1]} (norm)', fontsize=11)
    ax.set_ylabel(f'{param_names[idx2]} (norm)', fontsize=11)
    ax.set_title('(c) NN Surface', fontsize=12)
    
    plt.suptitle(f'Penalty Landscape: {penalty_name} (cf. Figure 3)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_posterior_histograms(samples, param_names, param_bounds, 
                               samples_no_unc=None, figsize=(16, 10)):
    """Posterior parameter histograms (Figure 10)."""
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        
        ax.hist(samples[:, i], bins=40, alpha=0.7, color='gold', 
                label='With obs unc.', density=True, edgecolor='orange')
        
        if samples_no_unc is not None:
            ax.hist(samples_no_unc[:, i], bins=40, alpha=0.5, color='steelblue',
                    label='Without obs unc.', density=True, edgecolor='navy')
        
        low, high = param_bounds[param_names[i]]
        ax.axvline(low, color='gray', linestyle='--', linewidth=2)
        ax.axvline(high, color='gray', linestyle='--', linewidth=2)
        
        ax.set_xlabel(param_names[i], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Posterior Parameter Distributions (cf. Figure 10)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_penalty_function_demo():
    """Visualize the penalty function (Figure 1g)."""
    obs_low, obs_high = 10, 15
    model_vals = np.linspace(5, 20, 100)
    penalties = compute_penalty(model_vals, obs_high, obs_low)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model_vals, penalties, 'g-', linewidth=3, label='Combined Penalty (Eq. 1)')
    ax.axvspan(obs_low, obs_high, alpha=0.2, color='green', label='No penalty zone')
    ax.axvline(obs_low, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(obs_high, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Model Value', fontsize=12)
    ax.set_ylabel('Penalty', fontsize=12)
    ax.set_title('Penalty Function: "Bowl with Flat Bottom" (cf. Figure 1g)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
