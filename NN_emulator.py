# =============================================================================
# NEURAL NETWORK EMULATOR FOLLOWING ELSAESSER ET AL. (2025)
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class NNEnsembleEmulator:
    """
    Neural Network ensemble emulator following Elsaesser et al. (2025).
    
    Key features:
    - Ensemble of 6 NNs with different architectures (Table 3 in paper)
    - Emulates PENALTY METRICS directly (not raw outputs)
    - Inputs normalized to [0, 1]
    - Outputs standardized to mean=0, std=1
    - Final prediction is ensemble average
    """
    
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Scalers (following paper: inputs 0-1, outputs standardized)
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = StandardScaler()
        
        # NN ensemble (will be populated in fit())
        self.nn_ensemble = []
        self.is_fitted = False
        
        # Store training history for diagnostics
        self.histories = []
        
    def _build_nn(self, architecture, activation='relu'):
        """
        Build a single NN with specified architecture.
        
        Parameters
        ----------
        architecture : list of int
            Number of nodes in each hidden layer
        activation : str
            Activation function ('relu' or 'leaky_relu')
        """
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.n_inputs,)))
        
        for n_nodes in architecture:
            if activation == 'leaky_relu':
                model.add(keras.layers.Dense(n_nodes))
                model.add(keras.layers.LeakyReLU(alpha=0.1))
            else:
                model.add(keras.layers.Dense(n_nodes, activation='relu'))
        
        # Output layer (linear activation for regression)
        model.add(keras.layers.Dense(self.n_outputs))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
    
    def _get_ensemble_architectures(self):
        """
        Return the 6 NN architectures from Table 3 of Elsaesser et al. (2025).
        """
        architectures = [
            {'layers': [64, 256, 256, 256], 'activation': 'relu'},      # NN01
            {'layers': [64, 64, 64], 'activation': 'relu'},              # NN02
            {'layers': [64, 128, 128], 'activation': 'relu'},            # NN04
            {'layers': [64, 64, 64], 'activation': 'leaky_relu'},        # NN05
            {'layers': [64, 64, 64, 64], 'activation': 'leaky_relu'},    # NN06
            {'layers': [64, 128, 256, 512], 'activation': 'relu'},       # NN11
        ]
        return architectures
    
    def fit(self, X, y, epochs=500, batch_size=32, validation_split=0.1, 
            verbose=0, early_stopping=True):
        """
        Fit the ensemble of NNs.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_inputs)
            Input parameters (physical parameters)
        y : array, shape (n_samples, n_outputs)
            Output penalties (already computed penalty metrics)
        """
        # Scale inputs to [0, 1]
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Standardize outputs to mean=0, std=1
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Build and train each NN in ensemble
        architectures = self._get_ensemble_architectures()
        self.nn_ensemble = []
        self.histories = []
        
        callbacks = []
        if early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50, restore_best_weights=True
            ))
        
        for i, arch in enumerate(architectures):
            print(f"Training NN {i+1}/6: {arch['layers']} ({arch['activation']})")
            
            model = self._build_nn(arch['layers'], arch['activation'])
            
            history = model.fit(
                X_scaled, y_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.nn_ensemble.append(model)
            self.histories.append(history)
        
        self.is_fitted = True
        print(f"Ensemble of {len(self.nn_ensemble)} NNs trained successfully.")
    
    def predict(self, X_new, return_std=False):
        """
        Predict using ensemble average.
        
        Parameters
        ----------
        X_new : array, shape (n_samples, n_inputs)
            New input parameters
        return_std : bool
            If True, also return ensemble standard deviation
            
        Returns
        -------
        mean_pred : array, shape (n_samples, n_outputs)
            Ensemble mean prediction (in original output scale)
        std_pred : array, optional
            Ensemble standard deviation (if return_std=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Emulator must be fitted first.")
        
        X_scaled = self.scaler_X.transform(X_new)
        
        # Get predictions from all NNs
        predictions_scaled = []
        for model in self.nn_ensemble:
            pred = model.predict(X_scaled, verbose=0)
            predictions_scaled.append(pred)
        
        predictions_scaled = np.array(predictions_scaled)  # (n_models, n_samples, n_outputs)
        
        # Ensemble average
        mean_scaled = np.mean(predictions_scaled, axis=0)
        
        # Inverse transform to original scale
        mean_pred = self.scaler_y.inverse_transform(mean_scaled)
        
        if return_std:
            std_scaled = np.std(predictions_scaled, axis=0)
            # Scale std back (multiply by scaler's scale)
            std_pred = std_scaled * self.scaler_y.scale_
            return mean_pred, std_pred
        
        return mean_pred
    
    def evaluate(self, X_test, y_test, output_names=None):
        """
        Evaluate emulator performance (like Figure 4 in paper).
        """
        y_pred = self.predict(X_test)
        
        results = {}
        n_outputs = y_test.shape[1]
        
        if output_names is None:
            output_names = [f'Output_{i}' for i in range(n_outputs)]
        
        print("\n" + "="*70)
        print("EMULATOR EVALUATION (cf. Figure 4 in Elsaesser et al.)")
        print("="*70)
        
        for i in range(n_outputs):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            results[output_names[i]] = {'rmse': rmse, 'r2': r2}
            print(f"{output_names[i]:30s}: R² = {r2:.3f}, RMSE = {rmse:.4f}")
        
        print("="*70)
        
        return results, y_pred


# =============================================================================
# PENALTY FUNCTION (Equation 1 from paper)
# =============================================================================

def compute_penalty(model_value, obs_high, obs_low, exponent=2):
    """
    Compute penalty following Equation 1 in Elsaesser et al. (2025).
    
    Key insight: If model falls within observational bounds, penalty = 0
    
    Parameters
    ----------
    model_value : float or array
        Model output value(s)
    obs_high : float
        Upper bound of observational estimate
    obs_low : float
        Lower bound of observational estimate
    exponent : int
        Exponent N in the penalty function (default 2)
        
    Returns
    -------
    penalty : float or array
        Penalty value (sqrt of accumulated squared penalties)
    """
    # Equation 1: (|M - O_H| + |M - O_L| - (O_H - O_L)) / 2
    # This equals 0 when O_L <= M <= O_H
    
    penalty_raw = (np.abs(model_value - obs_high) + 
                   np.abs(model_value - obs_low) - 
                   (obs_high - obs_low)) / 2.0
    
    # Ensure non-negative (should be by construction, but numerical safety)
    penalty_raw = np.maximum(penalty_raw, 0)
    
    # Apply exponent and return sqrt
    return np.sqrt(penalty_raw ** exponent)


def compute_all_penalties(stats_dict, obs_bounds, stat_names):
    """
    Compute penalty metrics for all statistics.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary of computed summary statistics
    obs_bounds : dict
        Dictionary with keys matching stat_names, values are (low, high) tuples
    stat_names : list
        Names of statistics to compute penalties for
        
    Returns
    -------
    penalties : array
        Array of penalty values for each statistic
    """
    penalties = []
    
    for name in stat_names:
        if name in stats_dict and name in obs_bounds:
            model_val = stats_dict[name]
            obs_low, obs_high = obs_bounds[name]
            
            # Handle scalars vs arrays
            if np.isscalar(model_val):
                penalty = compute_penalty(model_val, obs_high, obs_low)
            else:
                # For array outputs (like PCA components), compute element-wise
                penalty = compute_penalty(model_val, obs_high, obs_low)
            
            penalties.append(penalty)
        else:
            penalties.append(np.nan)
    
    return np.array(penalties)


# =============================================================================
# PREPARE PENALTY METRICS FOR YOUR DO DATA
# =============================================================================

def prepare_penalty_dataset(ensemble_stats, default_stats, obs_uncertainty_dict,
                           stat_names, has_DO_behavior):
    """
    Prepare penalty metrics dataset for NN training.
    
    Parameters
    ----------
    ensemble_stats : list of dicts
        Summary statistics for each ensemble member
    default_stats : dict
        Summary statistics for the target/default run
    obs_uncertainty_dict : dict
        Observational uncertainties for each statistic
        Format: {stat_name: uncertainty_value}
    stat_names : list
        Names of statistics to use as penalty metrics
    has_DO_behavior : array of bool
        Mask for valid DO-oscillating runs
        
    Returns
    -------
    penalty_matrix : array, shape (n_valid_runs, n_stats)
        Matrix of penalty values
    """
    penalties_list = []
    
    # Define observational bounds based on target ± uncertainty
    obs_bounds = {}
    for name in stat_names:
        if name in default_stats and name in obs_uncertainty_dict:
            target_val = default_stats[name]
            uncertainty = obs_uncertainty_dict[name]
            
            # Handle scalar vs array targets
            if np.isscalar(target_val):
                obs_bounds[name] = (target_val - uncertainty, target_val + uncertainty)
            else:
                # For arrays (like PCA components)
                obs_bounds[name] = (target_val - uncertainty, target_val + uncertainty)
    
    # Compute penalties for each valid run
    for i, stats in enumerate(ensemble_stats):
        if stats is not None and has_DO_behavior[i]:
            penalties = []
            
            for name in stat_names:
                if name in stats and name in obs_bounds:
                    model_val = stats[name]
                    obs_low, obs_high = obs_bounds[name]
                    
                    if np.isscalar(model_val):
                        penalty = compute_penalty(model_val, obs_high, obs_low)
                        penalties.append(penalty)
                    else:
                        # For array-valued stats (e.g., PCA components)
                        for j, val in enumerate(model_val):
                            low = obs_low[j] if hasattr(obs_low, '__len__') else obs_low
                            high = obs_high[j] if hasattr(obs_high, '__len__') else obs_high
                            penalty = compute_penalty(val, high, low)
                            penalties.append(penalty)
            
            penalties_list.append(penalties)
    
    return np.array(penalties_list), obs_bounds


# =============================================================================
# FULL PIPELINE FOR YOUR DO EMULATOR
# =============================================================================

def build_DO_emulator_elsaesser_style(X, ensemble_stats, default_stats, 
                                       has_DO_behavior, pca_model,
                                       obs_uncertainties, pdf_pca_components=5):
    """
    Build NN emulator following Elsaesser et al. methodology.
    
    Parameters
    ----------
    X : array, shape (n_runs, n_params)
        Input parameters for all runs
    ensemble_stats : list of dicts
        Summary statistics for each run
    default_stats : dict
        Target statistics (from default/observed run)
    has_DO_behavior : array of bool
        Mask for valid runs
    pca_model : sklearn PCA
        Fitted PCA model for PDFs
    obs_uncertainties : dict
        Observational uncertainties for each statistic
    pdf_pca_components : int
        Number of PCA components used
        
    Returns
    -------
    emulator : NNEnsembleEmulator
        Trained emulator
    penalty_matrix : array
        Computed penalties used for training
    """
    
    # Filter for valid runs
    X_valid = X[has_DO_behavior]
    
    # Define which statistics to use as penalty metrics
    scalar_stat_names = ['mean', 'std', 'avg_stadial_duration', 
                         'avg_waiting_time', 'avg_amplitude', 'n_stadials']
    
    # Build penalty matrix
    print("Computing penalty metrics...")
    
    penalty_list = []
    
    for i, (stats, valid) in enumerate(zip(ensemble_stats, has_DO_behavior)):
        if stats is not None and valid:
            run_penalties = []
            
            # PCA component penalties
            for j in range(pdf_pca_components):
                target = default_stats['pca_components'][j]
                model_val = stats['pca_components'][j]
                unc = obs_uncertainties.get(f'pca_{j}', 0.01)
                
                penalty = compute_penalty(model_val, target + unc, target - unc)
                run_penalties.append(penalty)
            
            # Scalar statistic penalties
            for name in scalar_stat_names:
                if name in stats and name in default_stats:
                    target = default_stats[name]
                    model_val = stats[name]
                    unc = obs_uncertainties.get(name, np.abs(target) * 0.1)  # 10% default
                    
                    penalty = compute_penalty(model_val, target + unc, target - unc)
                    run_penalties.append(penalty)
            
            penalty_list.append(run_penalties)
    
    penalty_matrix = np.array(penalty_list)
    
    # Build output names for diagnostics
    output_names = [f'PCA_{i+1}_penalty' for i in range(pdf_pca_components)]
    output_names += [f'{name}_penalty' for name in scalar_stat_names]
    
    print(f"Penalty matrix shape: {penalty_matrix.shape}")
    print(f"Input parameters shape: {X_valid.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, penalty_matrix, test_size=0.2, random_state=42
    )
    
    # Build and train emulator
    emulator = NNEnsembleEmulator(
        n_inputs=X_valid.shape[1],
        n_outputs=penalty_matrix.shape[1]
    )
    
    emulator.fit(X_train, y_train, epochs=500, verbose=0)
    
    # Evaluate
    results, y_pred = emulator.evaluate(X_test, y_test, output_names)
    
    return emulator, penalty_matrix, output_names, (X_train, X_test, y_train, y_test)



import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
import tensorflow as tf
from tensorflow import keras
import numpy as np


class NNEnsembleEmulatorSimple:
    """
    Simplified NN ensemble for smaller datasets.
    Uses smaller networks and more regularization.
    """
    
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = StandardScaler()
        self.nn_ensemble = []
        self.is_fitted = False
        
    def _build_nn(self, architecture, activation='relu', l2_reg=0.01):
        """Build NN with regularization."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.n_inputs,)))
        
        for n_nodes in architecture:
            model.add(keras.layers.Dense(
                n_nodes, 
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))
            model.add(keras.layers.Dropout(0.1))
        
        model.add(keras.layers.Dense(self.n_outputs))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        return model
    
    def _get_ensemble_architectures(self):
        """Simpler architectures for small datasets."""
        architectures = [
            {'layers': [32, 32], 'activation': 'relu'},
            {'layers': [64, 32], 'activation': 'relu'},
            {'layers': [32, 16], 'activation': 'relu'},
            {'layers': [64, 64], 'activation': 'relu'},
            {'layers': [32, 32, 16], 'activation': 'relu'},
            {'layers': [64], 'activation': 'relu'},
        ]
        return architectures
    
    def fit(self, X, y, epochs=1000, batch_size=32, validation_split=0.15, verbose=0):
        """Fit with early stopping."""
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        architectures = self._get_ensemble_architectures()
        self.nn_ensemble = []
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=100,  # More patience
                restore_best_weights=True,
                min_delta=1e-6
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=50,
                min_lr=1e-6
            )
        ]
        
        for i, arch in enumerate(architectures):
            print(f"  Training NN {i+1}/6: {arch['layers']}")
            
            model = self._build_nn(arch['layers'], arch['activation'])
            
            history = model.fit(
                X_scaled, y_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.nn_ensemble.append(model)
            
            # Report final performance
            val_loss = min(history.history['val_loss'])
            n_epochs = len(history.history['loss'])
            print(f"    Stopped at epoch {n_epochs}, best val_loss: {val_loss:.6f}")
        
        self.is_fitted = True
        print(f"\n  Ensemble trained.")
    
    def predict(self, X_new, return_std=False, return_all=False):
        """Predict with ensemble average."""
        if not self.is_fitted:
            raise RuntimeError("Must fit first")
        
        X_scaled = self.scaler_X.transform(X_new)
        
        predictions_scaled = []
        for model in self.nn_ensemble:
            pred = model.predict(X_scaled, verbose=0)
            predictions_scaled.append(pred)
        
        predictions_scaled = np.array(predictions_scaled)
        mean_scaled = np.mean(predictions_scaled, axis=0)
        mean_pred = self.scaler_y.inverse_transform(mean_scaled)
        
        if return_all:
            all_preds = []
            for pred in predictions_scaled:
                all_preds.append(self.scaler_y.inverse_transform(pred))
            return mean_pred, np.array(all_preds)
        
        if return_std:
            std_scaled = np.std(predictions_scaled, axis=0)
            std_pred = std_scaled * self.scaler_y.scale_
            return mean_pred, std_pred
        
        return mean_pred