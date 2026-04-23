import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from scipy import signal
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def lowpass_filter(data, cutoff_years, dt=1.0):
    """Butterworth lowpass filter with forward-backward pass (no phase shift)."""
    cutoff_freq = 1.0 / cutoff_years
    nyquist_freq = 1.0 / (2.0 * dt)
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


# ---------------------------------------------------------------------------
# AMOC time series & PDF visualisation
# ---------------------------------------------------------------------------

def plot_amoc_analysis(ensemble_stats, model_files, default_stats=None, default_file=None,
                       run_ids=None, n_runs=None, spinup_fraction=0.02,
                       figsize_per_row=3, width_ratios=(3, 2),
                       show_default=True, save_path=None):
    """
    Plot AMOC time series and KDE PDFs for selected ensemble members.

    Parameters
    ----------
    ensemble_stats : list of dicts from compute_summary_stats()
    model_files    : list of file paths (None for missing runs)
    default_stats  : stats dict for the default/target run
    default_file   : path to default run NetCDF
    run_ids        : list of specific run indices to plot (0-based); overrides n_runs
    n_runs         : number of runs to plot from the start; ignored if run_ids given
    spinup_fraction: fraction removed from the start as spinup
    show_default   : prepend a row for the default run
    save_path      : if given, save figure to this path

    Returns
    -------
    fig, axes
    """
    if run_ids is not None:
        runs_to_plot = list(run_ids)
    elif n_runs is not None:
        runs_to_plot = list(range(n_runs))
    else:
        runs_to_plot = list(range(min(10, len(ensemble_stats))))

    runs_to_plot = [i for i in runs_to_plot
                    if i < len(ensemble_stats) and ensemble_stats[i] is not None]

    if not runs_to_plot:
        raise ValueError("No valid runs to plot.")

    n_rows = len(runs_to_plot) + (1 if show_default and default_stats is not None else 0)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(18, figsize_per_row * n_rows),
                             gridspec_kw={'width_ratios': list(width_ratios)})
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    row = 0
    if show_default and default_stats is not None and default_file is not None:
        ds = xr.open_dataset(default_file)
        amoc = ds.amoc26N.values
        time = ds.time.values
        ds.close()
        s = int(len(amoc) * 0.02)
        _plot_single_run(axes[row], time[s:] - time[s], amoc[s:], default_stats,
                         title='AMOC — Default run')
        row += 1

    for idx in runs_to_plot:
        file = model_files[idx]
        ds = xr.open_dataset(file)
        amoc = ds.amoc26N.values
        time = ds.time.values
        ds.close()
        s = int(len(amoc) * spinup_fraction)
        _plot_single_run(axes[row], time[s:] - time[s], amoc[s:], ensemble_stats[idx],
                         title=f'AMOC — Run {idx + 1}')
        row += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


def _plot_single_run(ax_row, time, amoc, stats, title='AMOC'):
    """Plot one AMOC time series (left) and its KDE PDF (right)."""
    ax_ts, ax_pdf = ax_row

    ax_ts.plot(time, amoc, color='darkblue', lw=0.8, label='AMOC')
    ax_ts.axhline(stats['threshold'], color='gray', ls='--', alpha=0.5, label='Threshold')
    for t in stats.get('do_times', []):
        ax_ts.axvline(t, color='red', alpha=0.5, lw=1.2)

    do_var = stats.get('do_variability', True)
    box_color = 'wheat' if do_var else 'lightsalmon'
    ax_ts.text(
        0.98, 0.02,
        f"DO variability: {'Yes' if do_var else 'No'}\n"
        f"N DO events: {stats['n_do_events']}\n"
        f"Avg waiting time: {stats['avg_waiting_time']:.1f} yr\n"
        f"Avg stadial duration: {stats['avg_stadial_duration']:.1f} yr\n"
        f"Amplitude: {stats['avg_amplitude']:.2f} Sv",
        transform=ax_ts.transAxes, fontsize=9,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
    )
    ax_ts.set_ylabel('AMOC (Sv)')
    ax_ts.set_title(title)
    ax_ts.legend(loc='upper left', fontsize=8)
    ax_ts.grid(alpha=0.3)

    ax_pdf.plot(stats['x_grid'], stats['pdf'], color='darkblue')
    ax_pdf.axvline(stats['threshold'], color='gray', ls='--', alpha=0.5)
    ax_pdf.set_xlabel('AMOC (Sv)')
    ax_pdf.set_ylabel('Density')
    ax_pdf.grid(alpha=0.3)


def plot_filtered_runs(ensemble_stats, model_files, filter_mask,
                       default_stats=None, default_file=None, max_runs=20, **kwargs):
    """Plot ensemble runs selected by a boolean mask."""
    run_ids = np.where(filter_mask)[0][:max_runs]
    print(f"Plotting {len(run_ids)} / {filter_mask.sum()} filtered runs")
    return plot_amoc_analysis(ensemble_stats, model_files,
                              default_stats=default_stats, default_file=default_file,
                              run_ids=run_ids, **kwargs)


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def plot_pca_variance(pca_model, figsize=(7, 4)):
    """Bar chart of explained variance ratio per PCA component."""
    ratios = pca_model.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n = len(ratios)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(1, n + 1), ratios * 100, color='steelblue', alpha=0.8, label='Individual')
    ax.plot(range(1, n + 1), cum * 100, 'o-', color='darkorange', lw=2, label='Cumulative')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA of AMOC PDF — Explained Variance')
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pca_components(pca_model, x_grid, n_components=None, figsize=None):
    """Plot PCA eigenvectors (principal components) as functions of AMOC."""
    n = n_components or pca_model.n_components_
    n_cols = min(n, 3)
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (5 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n):
        axes[i].plot(x_grid, pca_model.components_[i], color='darkblue', lw=2)
        axes[i].axhline(0, color='gray', ls='--', alpha=0.5)
        axes[i].set_title(f'PC {i + 1}  ({pca_model.explained_variance_ratio_[i]*100:.1f}%)')
        axes[i].set_xlabel('AMOC (Sv)')
        axes[i].grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('PCA Components of AMOC PDF', fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Emulator evaluation plots
# ---------------------------------------------------------------------------

def plot_pca_predictions(pca_test, pca_pred, pca_std=None, figsize=None):
    """
    Scatter plots of true vs predicted PCA scores for each component.

    Parameters
    ----------
    pca_test, pca_pred : (n, n_pca)
    pca_std            : (n, n_pca) optional predictive std for error bars
    """
    n = pca_test.shape[1]
    n_cols = min(n, 3)
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        true, pred = pca_test[:, i], pca_pred[:, i]

        ax.scatter(true, pred, alpha=0.5, s=25, color='steelblue')

        if pca_std is not None:
            sample_idx = np.random.choice(len(true), min(30, len(true)), replace=False)
            ax.errorbar(true[sample_idx], pred[sample_idx], yerr=pca_std[sample_idx, i],
                        fmt='none', ecolor='gray', alpha=0.4)

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)

        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        ax.set_title(f'PCA {i+1}  R²={r2:.3f}  RMSE={rmse:.4f}')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('GP Emulator: Predicted vs True PCA Scores', fontsize=13)
    plt.tight_layout()
    return fig


def plot_stat_predictions(Y_test, Y_pred, stat_names, figsize=None):
    """Scatter plots of true vs predicted scalar statistics."""
    n = Y_test.shape[1]
    n_cols = min(n, 3)
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        true, pred = Y_test[:, i], Y_pred[:, i]
        ax.scatter(true, pred, alpha=0.5, s=25, color='steelblue')
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        name = stat_names[i] if i < len(stat_names) else f'stat_{i}'
        ax.set_title(f'{name}  R²={r2:.3f}  RMSE={rmse:.4f}')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('GP Emulator: Predicted vs True Scalar Statistics', fontsize=13)
    plt.tight_layout()
    return fig


def plot_pdf_comparison(pdf_true, pdf_pred, x_grid, n_examples=6, figsize=(14, 8)):
    """Plot overlaid true and predicted PDFs for a random subset of test samples."""
    n = pdf_true.shape[0]
    idx = np.linspace(0, n - 1, n_examples, dtype=int)

    n_cols = 3
    n_rows = int(np.ceil(n_examples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for k, i in enumerate(idx):
        ax = axes[k]
        rmse = np.sqrt(np.mean((pdf_true[i] - pdf_pred[i]) ** 2))
        ax.plot(x_grid, pdf_true[i], 'b-', lw=2, label='True')
        ax.plot(x_grid, pdf_pred[i], 'r--', lw=2, label='Predicted')
        ax.set_title(f'Sample {i}  RMSE={rmse:.4f}')
        ax.set_xlabel('AMOC (Sv)')
        ax.set_ylabel('PDF')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(n_examples, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('PDF Reconstruction Examples', fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parameter-output correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(X, Y, param_names, output_names, title='Parameter–Output Correlations',
                             figsize=(14, 8)):
    """Pearson correlation heatmap between input parameters and emulator outputs."""
    n_params = X.shape[1]
    n_outputs = Y.shape[1]
    corr = np.zeros((n_outputs, n_params))
    for i in range(n_outputs):
        for j in range(n_params):
            c = np.corrcoef(Y[:, i], X[:, j])[0, 1]
            corr[i, j] = c if np.isfinite(c) else 0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_outputs))
    ax.set_yticklabels(output_names, fontsize=10)
    ax.set_title(title, fontsize=13)

    for i in range(n_outputs):
        for j in range(n_params):
            if abs(corr[i, j]) > 0.3:
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black')

    plt.tight_layout()
    return fig
