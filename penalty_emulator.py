import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =============================================================================
# OBSERVATIONAL UNCERTAINTIES (from your sliding window analysis)
# =============================================================================

# PCA component uncertainties
SIGMA_PCA_OBS = np.array([0.01, 0.005, 0.01, 0.003, 0.005])  # std for PCA 1-5

# Dynamical statistics uncertainties
SIGMA_WAITING_OBS = 42.36    # years
SIGMA_STADIAL_OBS = 34.94    # years
SIGMA_MEAN_OBS = 0.03        # Sv
SIGMA_STD_OBS = 0.03         # Sv
SIGMA_AMPLITUDE_OBS = 0.68   # Sv
SIGMA_N_DO_OBS = 1.28        # count
SIGMA_N_STADIALS_OBS = 1.86  # count

# =============================================================================
# PENALTY FUNCTION DEFINITIONS
# =============================================================================

def compute_penalty_with_uncertainty(model_value, obs_center, obs_sigma, n_sigma=2):
    """
    Compute penalty using absolute uncertainty (sigma).
    
    If model falls within [obs_center - n_sigma*sigma, obs_center + n_sigma*sigma], 
    penalty contribution is 0 (or reduced).
    
    This follows the spirit of Elsaesser et al. (2025) Equation 1,
    but uses symmetric sigma-based bounds.
    
    Parameters
    ----------
    model_value : float or array
        Model output value(s)
    obs_center : float
        Observational central estimate
    obs_sigma : float
        Observational uncertainty (1-sigma)
    n_sigma : float
        Number of sigmas for the "no penalty" zone (default 2 for ~95% CI)
    
    Returns
    -------
    penalty : float or array
        Penalty value (0 if within bounds)
    """
    obs_low = obs_center - n_sigma * obs_sigma
    obs_high = obs_center + n_sigma * obs_sigma
    
    # Equation 1 from paper adapted:
    # C = ((|M - O_H| + |M - O_L| - (O_H - O_L)) / 2)^N
    diff_high = np.abs(model_value - obs_high)
    diff_low = np.abs(model_value - obs_low)
    obs_range = obs_high - obs_low
    
    # This gives 0 when model is within [obs_low, obs_high]
    penalty_raw = (diff_high + diff_low - obs_range) / 2
    penalty_raw = np.maximum(penalty_raw, 0)
    
    # Normalize by sigma to make penalties comparable across metrics
    penalty_normalized = penalty_raw / obs_sigma
    
    return penalty_normalized


def compute_penalty_simple(model_value, obs_center, obs_sigma):
    """
    Simple normalized squared error penalty (no flat region).
    
    Useful when you want continuous gradient for optimization.
    """
    return ((model_value - obs_center) / obs_sigma) ** 2


# =============================================================================
# OBSERVATIONAL TARGETS CLASS
# =============================================================================

class ObservationalTargets:
    """
    Container for observational targets with absolute uncertainties.
    """
    
    def __init__(self):
        self.targets = {}
        
    def add_target(self, name, center, sigma, weight=1.0, n_sigma=2):
        """
        Add an observational target with absolute uncertainty.
        
        Parameters
        ----------
        name : str
            Name of the metric
        center : float or array
            Central observational estimate
        sigma : float or array
            Absolute uncertainty (1-sigma)
        weight : float
            Relative importance weight
        n_sigma : float
            Width of "no penalty" zone in units of sigma
        """
        self.targets[name] = {
            'center': np.atleast_1d(center),
            'sigma': np.atleast_1d(sigma),
            'low': np.atleast_1d(center) - n_sigma * np.atleast_1d(sigma),
            'high': np.atleast_1d(center) + n_sigma * np.atleast_1d(sigma),
            'weight': weight,
            'n_sigma': n_sigma,
            'is_vector': isinstance(center, np.ndarray) and len(center) > 1
        }
    
    def get_target(self, name):
        return self.targets.get(name, None)
    
    def list_targets(self):
        return list(self.targets.keys())
    
    def print_summary(self):
        print("\nObservational Targets Summary:")
        print("-" * 60)
        for name, t in self.targets.items():
            if t['is_vector']:
                print(f"  {name}: {len(t['center'])} components")
                for i in range(len(t['center'])):
                    print(f"    [{i}]: {t['center'][i]:.4f} ± {t['sigma'][i]:.4f} "
                          f"(±{t['n_sigma']}σ: [{t['low'][i]:.4f}, {t['high'][i]:.4f}])")
            else:
                print(f"  {name}: {t['center'][0]:.4f} ± {t['sigma'][0]:.4f} "
                      f"(±{t['n_sigma']}σ: [{t['low'][0]:.4f}, {t['high'][0]:.4f}])")


# =============================================================================
# PENALTY CALCULATOR
# =============================================================================

class PenaltyCalculator:
    """
    Computes penalty metrics for model outputs given observational targets.
    """
    
    def __init__(self, obs_targets, use_flat_penalty=True):
        """
        Parameters
        ----------
        obs_targets : ObservationalTargets
            Observational targets with uncertainty bounds
        use_flat_penalty : bool
            If True, use penalty function with flat region within uncertainty bounds.
            If False, use simple normalized squared error (continuous gradient).
        """
        self.obs_targets = obs_targets
        self.use_flat_penalty = use_flat_penalty
        
        # Build list of penalty names
        self.penalty_names = []
        for name, target in obs_targets.targets.items():
            if target['is_vector']:
                # Vector target (like PCA components) - one penalty per component
                for i in range(len(target['center'])):
                    self.penalty_names.append(f'penalty_{name}_{i}')
            else:
                self.penalty_names.append(f'penalty_{name}')
    
    def compute_penalties(self, model_values):
        """
        Compute all penalty metrics for a single model run.
        
        Parameters
        ----------
        model_values : dict
            Dictionary with model statistics matching target names
            e.g., {'pca_components': array, 'avg_waiting_time': float, ...}
            
        Returns
        -------
        penalties : array
            Array of penalty values in same order as self.penalty_names
        """
        penalties = []
        
        for name, target in self.obs_targets.targets.items():
            model_val = model_values.get(name)
            if model_val is None:
                # Skip if model value not provided
                if target['is_vector']:
                    penalties.extend([np.nan] * len(target['center']))
                else:
                    penalties.append(np.nan)
                continue
            
            model_val = np.atleast_1d(model_val)
            
            if self.use_flat_penalty:
                # Penalty with flat region within uncertainty bounds
                pen = compute_penalty_with_uncertainty(
                    model_val, 
                    target['center'], 
                    target['sigma'],
                    target['n_sigma']
                )
            else:
                # Simple normalized squared error
                pen = compute_penalty_simple(
                    model_val,
                    target['center'],
                    target['sigma']
                )
            
            # Apply weight
            pen = pen * target['weight']
            
            penalties.extend(pen.flatten())
        
        return np.array(penalties)
    
    def compute_penalties_batch(self, model_values_list):
        """
        Compute penalties for multiple model runs.
        
        Parameters
        ----------
        model_values_list : list of dicts
            Each dict contains model statistics for one run
        
        Returns
        -------
        penalty_matrix : array, shape (n_runs, n_penalties)
        """
        n_runs = len(model_values_list)
        n_penalties = len(self.penalty_names)
        
        penalty_matrix = np.zeros((n_runs, n_penalties))
        
        for i, model_values in enumerate(model_values_list):
            penalty_matrix[i] = self.compute_penalties(model_values)
        
        return penalty_matrix


# =============================================================================
# PENALTY-BASED GP EMULATOR
# =============================================================================

class GPEmulatorPenalty:
    """
    GP emulator that predicts penalty metrics directly.
    """
    
    def __init__(self, kernel=None, penalty_names=None):
        self.kernel = kernel if kernel is not None else (
            ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + 
            WhiteKernel(noise_level=1e-5)
        )
        self.scaler_X = StandardScaler()
        self.scaler_penalties = StandardScaler()
        self.gp_model = None
        self.is_fitted = False
        self.penalty_names = penalty_names
        
    def fit(self, X, penalties):
        """
        Fit the GP emulator on penalty metrics.
        """
        self.n_penalties = penalties.shape[1]
        
        X_scaled = self.scaler_X.fit_transform(X)
        penalties_scaled = self.scaler_penalties.fit_transform(penalties)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            alpha=1e-6
        )
        self.gp_model.fit(X_scaled, penalties_scaled)
        self.is_fitted = True
        
        print(f"Fitted GP emulator for {self.n_penalties} penalty metrics")
        
    def predict(self, X_new, return_std=True):
        """
        Predict penalty metrics for new parameter values.
        """
        if not self.is_fitted:
            raise RuntimeError("Emulator must be fitted first")
        
        X_scaled = self.scaler_X.transform(X_new)
        
        if return_std:
            mean_scaled, std_scaled = self.gp_model.predict(X_scaled, return_std=True)
            mean = self.scaler_penalties.inverse_transform(mean_scaled)
            # Handle std properly for multi-output
            if mean_scaled.ndim == 1:
                mean_scaled = mean_scaled.reshape(1, -1)
            std = std_scaled.reshape(-1, 1) * self.scaler_penalties.scale_
            return mean, std
        else:
            mean_scaled = self.gp_model.predict(X_scaled)
            mean = self.scaler_penalties.inverse_transform(mean_scaled)
            return mean
    
    def compute_total_penalty(self, X_new, weights=None):
        """
        Compute weighted total penalty for parameter inference.
        """
        penalties = self.predict(X_new, return_std=False)
        
        if weights is None:
            weights = np.ones(self.n_penalties)
        
        # Sum of squared penalties
        total = np.sum(weights * penalties**2, axis=1)
        
        return np.sqrt(total)


# =============================================================================
# SETUP FUNCTIONS FOR YOUR DATA
# =============================================================================

def setup_DO_targets(default_stats, n_sigma=2):
    """
    Set up observational targets using your sliding window uncertainties.
    
    Parameters
    ----------
    default_stats : dict
        Statistics from your default/reference run
    n_sigma : float
        Width of "no penalty" zone (2 = ~95% CI)
    
    Returns
    -------
    obs_targets : ObservationalTargets
    """
    obs_targets = ObservationalTargets()
    
    # PCA components (vector target)
    obs_targets.add_target(
        'pca_components',
        center=default_stats['pca_components'],
        sigma=SIGMA_PCA_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    # Scalar statistics
    obs_targets.add_target(
        'avg_waiting_time',
        center=default_stats['avg_waiting_time'],
        sigma=SIGMA_WAITING_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    obs_targets.add_target(
        'avg_stadial_duration',
        center=default_stats['avg_stadial_duration'],
        sigma=SIGMA_STADIAL_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    obs_targets.add_target(
        'mean',
        center=default_stats['mean'],
        sigma=SIGMA_MEAN_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    obs_targets.add_target(
        'std',
        center=default_stats['std'],
        sigma=SIGMA_STD_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    obs_targets.add_target(
        'avg_amplitude',
        center=default_stats['avg_amplitude'],
        sigma=SIGMA_AMPLITUDE_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    obs_targets.add_target(
        'n_stadials',
        center=default_stats['n_stadials'],
        sigma=SIGMA_N_STADIALS_OBS,
        weight=1.0,
        n_sigma=n_sigma
    )
    
    return obs_targets


def prepare_model_values_list(ensemble_stats_list, pca_components_array):
    """
    Convert your ensemble stats to the format expected by PenaltyCalculator.
    
    Parameters
    ----------
    ensemble_stats_list : list of dicts
        Each dict contains computed statistics for one run
    pca_components_array : array, shape (n_runs, n_pca_components)
        Pre-computed PCA components
    
    Returns
    -------
    model_values_list : list of dicts
        Formatted for PenaltyCalculator
    """
    model_values_list = []
    
    for i, stats in enumerate(ensemble_stats_list):
        model_values = {
            'pca_components': pca_components_array[i],
            'avg_waiting_time': stats['avg_waiting_time'],
            'avg_stadial_duration': stats['avg_stadial_duration'],
            'mean': stats['mean'],
            'std': stats['std'],
            'avg_amplitude': stats['avg_amplitude'],
            'n_stadials': stats['n_stadials']
        }
        model_values_list.append(model_values)
    
    return model_values_list