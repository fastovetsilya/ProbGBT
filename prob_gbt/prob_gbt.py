import numpy as np
import pandas as pd
import os
import tarfile
import json
import tempfile
import shutil
import warnings
import sys
import contextlib
from catboost import CatBoostRegressor
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from pygam import LinearGAM, s
from tqdm import tqdm
import properscoring as ps

# Global warning suppression for pygam convergence warnings
# This helps ensure warnings are suppressed everywhere in the code
warnings.filterwarnings("ignore", message=".*[Dd]id not converge.*")
warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygam")

# Context manager to completely silence stdout and stderr
class SilenceOutput:
    """
    Context manager to completely suppress all stdout and stderr output.
    This is used to silence PyGAM's output that may bypass warning filters.
    """
    def __init__(self):
        # Open null devices
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual file descriptors
        self.save_fds = [os.dup(1), os.dup(2)]
        
    def __enter__(self):
        # Assign the null pointers to stdout and stderr
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        
    def __exit__(self, *args):
        # Re-assign the original file descriptors to stdout and stderr
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class ProbGBT:
    """
    Probabilistic Gradient Boosted Trees model that provides uncertainty estimates.
    
    This model uses CatBoost's MultiQuantile loss function to predict multiple quantiles
    of the target distribution, then constructs a probability density function (PDF)
    from these quantiles.
    """
    
    def __init__(self, 
                 num_quantiles=50, 
                 iterations=500,
                 learning_rate=None, 
                 depth=None,
                 subsample=1.0,
                 random_seed=42,
                 train_separate_models=False):
        """
        Initialize the ProbGBT model.
        
        Parameters:
        -----------
        num_quantiles : int, default=50
            Number of quantiles to predict.
        iterations : int, default=500
            Maximum number of trees to build.
        learning_rate : float, optional
            Learning rate for the gradient boosting algorithm.
        depth : int, optional
            Depth of the trees.
        subsample : float, default=1.0
            Subsample ratio of the training instances.
        random_seed : int, default=42
            Random seed for reproducibility.
        train_separate_models : bool, default=False
            If True, train separate models for each quantile instead of using MultiQuantile loss.
        """
        self.num_quantiles = num_quantiles
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.subsample = subsample
        self.random_seed = random_seed
        self.train_separate_models = train_separate_models
        self.model = None
        self.trained_models = {}
        self.quantiles = None
        
    def _generate_non_uniform_quantiles(self):
        """
        Generate non-uniformly spaced quantiles with more focus on the tails.
        
        Returns:
        --------
        numpy.ndarray
            Array of quantile values between 0 and 1.
        """
        # Uniformly spaced quantiles
        uniform_quantiles = np.linspace(0.01, 0.99, self.num_quantiles)
        
        # Transform using normal distribution's PPF and CDF to focus more on the tails
        non_uniform_quantiles = norm.cdf(norm.ppf(uniform_quantiles) * 3)
        
        # Ensure values are within [0,1]
        non_uniform_quantiles = np.clip(non_uniform_quantiles, 0, 1)
        
        return non_uniform_quantiles
    
    def _smooth_quantile_function(self, quantiles, predictions, num_points=1000):
        """
        Create smoothed quantile function, CDF and PDF from discrete quantiles.
        
        Parameters:
        -----------
        quantiles : numpy.ndarray
            Quantile values (between 0 and 1).
        predictions : numpy.ndarray
            Predicted values for each quantile.
        num_points : int, default=1000
            Number of points to use for smoothing.
            
        Returns:
        --------
        tuple:
            (smoothed_x, smoothed_y, pdf_values) where:
            - smoothed_x: x-values for the smoothed function (predicted values)
            - smoothed_y: y-values for the smoothed function (quantiles)
            - pdf_values: PDF values corresponding to smoothed_x
        """
        # Fit a GAM to smooth the quantile function - disable verbose output
        gam = LinearGAM(s(0, constraints="monotonic_inc"), verbose=False).fit(quantiles, predictions)
        
        # Generate smoothed CDF
        quantiles_smooth = np.linspace(0, 1, num_points)
        y_pred_smooth = gam.predict(quantiles_smooth)
        
        # Compute PDF (derivative of the quantile function)
        epsilon = 1e-10  # Small value to avoid division by zero
        pdf_smooth = np.gradient(quantiles_smooth, y_pred_smooth + epsilon)
        pdf_smooth = np.maximum(pdf_smooth, 0)  # Ensure non-negative
        
        # Normalize the PDF
        total_area = np.trapz(pdf_smooth, y_pred_smooth)
        if total_area > epsilon:
            pdf_smooth /= total_area
        
        return y_pred_smooth, quantiles_smooth, pdf_smooth
    
    def _compute_crps(self, y_val, y_val_pred, num_points=1000, verbose=True):
        """
        Compute the Continuous Ranked Probability Score (CRPS) for validation data.
        
        Parameters:
        -----------
        y_val : numpy.ndarray
            Validation target values.
        y_val_pred : numpy.ndarray
            Validation predictions (quantiles) with shape (n_samples, n_quantiles).
        num_points : int, default=10000
            Number of points to use for smoothing.
        verbose : bool, default=True
            If True, print warning messages for calculation errors.
            
        Returns:
        --------
        float
            Mean CRPS value across validation samples.
        """
        crps_all = []
        
        # Suppress all warnings and stdout/stderr during CRPS computation
        with warnings.catch_warnings(), SilenceOutput():
            warnings.simplefilter("ignore")
            
            for i in range(len(y_val)):
                try:
                    y_sample = y_val[i]
                    y_pred_sample = y_val_pred[i]
                    
                    # Advanced approach with smoothing - compute PDF and use as weights
                    # Use the helper method to get smoothed values and PDF
                    y_pred_smooth, _, pdf_smooth = self._smooth_quantile_function(
                        self.quantiles, y_pred_sample, num_points=num_points
                    )
                    
                    # Use the smoothed prediction points and PDF as weights for CRPS
                    crps = ps.crps_ensemble(y_sample, y_pred_smooth, weights=pdf_smooth)
                    crps_all.append(crps)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Error calculating CRPS for sample {i}: {str(e)}")
                    continue
        
        # Filter out NaN and Inf values before computing mean
        crps_all = np.array(crps_all)
        valid_crps = crps_all[~np.isnan(crps_all) & ~np.isinf(crps_all)]
        
        if len(valid_crps) == 0:
            if verbose:
                print("Warning: No valid CRPS values computed!")
            return 999999.0
            
        return np.mean(valid_crps)

    def train(self, X, y, cat_features=None, eval_set=None, use_best_model=False, verbose=True, early_stopping_rounds=None):
        """
        Train the ProbGBT model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Training features.
        y : numpy.ndarray
            Target values.
        cat_features : list, optional
            List of categorical feature indices or names.
        eval_set : tuple, optional
            (X_val, y_val) for validation during training.
        use_best_model : bool, default=False
            If True, use the best model found based on the validation set.
            If False, the model will not be shrunk to the best iteration and will retain all trained iterations.
        verbose : bool, default=True
            If True, print training progress.
        early_stopping_rounds : int, optional
            Activates early stopping. Validation score needs to increase for this many rounds.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Generate quantiles
        self.quantiles = self._generate_non_uniform_quantiles()
        
        if self.train_separate_models:
            # Train separate models for each quantile
            self.trained_models = {}
            num_iter = self.iterations
            
            for q in tqdm(self.quantiles, disable=not verbose):
                # Define quantile loss function
                quantile_loss = f'Quantile:alpha={q}'
                
                # Create CatBoostRegressor model
                model = CatBoostRegressor(
                    cat_features=cat_features,
                    loss_function=quantile_loss,
                    iterations=num_iter,
                    learning_rate=self.learning_rate,
                    depth=self.depth,
                    subsample=self.subsample,
                    random_seed=self.random_seed
                )
                
                # Train the model
                model.fit(X, y, eval_set=eval_set, use_best_model=use_best_model, verbose=False,
                          early_stopping_rounds=early_stopping_rounds)
                
                # Add the model to the trained_models dictionary
                self.trained_models[q] = model
        else:
            # Format quantiles for CatBoost
            multiquantile_loss_str = str(self.quantiles.astype(float)).replace("[", "").replace("]", "").replace("\n", "").replace(" ", ", ")
            
            # Create CatBoost model parameters
            params = {
                'cat_features': cat_features,
                'loss_function': f"MultiQuantile:alpha={multiquantile_loss_str}",
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'subsample': self.subsample,
                'random_seed': self.random_seed,
                'verbose': verbose  # Show metrics every iteration
            }
            
            # Standard training - removed custom metrics
            self.model = CatBoostRegressor(**params)
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                use_best_model=use_best_model,
                verbose=verbose,  # Set verbose to boolean value
                early_stopping_rounds=early_stopping_rounds
            )
            
            # Display metrics from training
            if verbose:
                try:
                    metrics = self.model.get_evals_result()
                    print("\nTraining metrics:")
                    for dataset in metrics:
                        print(f"Dataset: {dataset}")
                        for metric_name, values in metrics[dataset].items():
                            print(f"  {metric_name}: {len(values)} evaluations")
                            # Show some sample values
                            if len(values) > 0:
                                print(f"    First value: {values[0]}")
                                print(f"    Last value: {values[-1]}")
                except Exception as e:
                    print(f"Could not display metrics: {str(e)}")
        
        return self
    
    def predict(self, X, return_quantiles=False):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        return_quantiles : bool, default=False
            If True, return the raw quantile predictions.
            
        Returns:
        --------
        If return_quantiles=True:
            numpy.ndarray: Raw quantile predictions with shape (n_samples, n_quantiles)
        If return_quantiles=False:
            numpy.ndarray: Mean predictions with shape (n_samples,)
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
        if self.train_separate_models:
            # Convert X to DataFrame if it's not already
            if not isinstance(X, pd.DataFrame):
                if len(X.shape) == 1:
                    X = pd.DataFrame([X])
                else:
                    X = pd.DataFrame(X)
            
            # Get predictions for each quantile
            quantile_preds = []
            for i in range(len(X)):
                sample_preds = []
                # Create a single-row DataFrame for this sample to preserve categorical features
                sample_df = X.iloc[[i]]
                
                for q in self.quantiles:
                    # Predict using the DataFrame directly instead of reshaping to numpy array
                    sample_preds.append(self.trained_models[q].predict(sample_df)[0])
                quantile_preds.append(sample_preds)
            quantile_preds = np.array(quantile_preds)
        else:
            # Get quantile predictions from the single model
            quantile_preds = self.model.predict(X)
        
        if return_quantiles:
            return quantile_preds
        
        # Return mean prediction (median)
        median_idx = np.searchsorted(self.quantiles, 0.5)
        if median_idx >= len(self.quantiles):
            median_idx = len(self.quantiles) - 1
        
        # If quantile_preds is 2D, extract the median for each sample
        if len(quantile_preds.shape) > 1:
            return quantile_preds[:, median_idx]
        else:
            # For a single sample
            return quantile_preds[median_idx]
    
    def predict_interval(self, X, confidence_level=0.95):
        """
        Predict confidence intervals for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        confidence_level : float, default=0.95
            Confidence level for the interval (between 0 and 1).
            
        Returns:
        --------
        tuple: (lower_bounds, upper_bounds) arrays for the confidence intervals
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
        quantile_preds = self.predict(X, return_quantiles=True)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        lower_bounds = []
        upper_bounds = []
        
        # Suppress all warnings and stdout/stderr during interval computation
        with warnings.catch_warnings(), SilenceOutput():
            warnings.simplefilter("ignore")
            
            for i in range(quantile_preds.shape[0]):
                # Get the predicted quantiles for this sample
                y_pred_sample = quantile_preds[i]
                
                # Use the helper method to get smoothed values
                y_pred_smooth, quantiles_smooth, _ = self._smooth_quantile_function(
                    self.quantiles, y_pred_sample, num_points=1000
                )
                
                # Find the quantile values corresponding to the confidence interval
                lower_idx = np.searchsorted(quantiles_smooth, (1 - confidence_level) / 2)
                upper_idx = np.searchsorted(quantiles_smooth, 1 - (1 - confidence_level) / 2)
                
                lower_bounds.append(y_pred_smooth[lower_idx])
                upper_bounds.append(y_pred_smooth[upper_idx])
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def predict_pdf(self, X, num_points=1000):
        """
        Predict the probability density function for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        num_points : int, default=1000
            Number of points to use for the PDF.
            
        Returns:
        --------
        list of tuples: [(x_values, pdf_values), ...] for each sample
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
        quantile_preds = self.predict(X, return_quantiles=True)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        results = []
        
        # Suppress all warnings and stdout/stderr during PDF computation
        with warnings.catch_warnings(), SilenceOutput():
            warnings.simplefilter("ignore")
            
            for i in range(quantile_preds.shape[0]):
                # Get the predicted quantiles for this sample
                y_pred_sample = quantile_preds[i]
                
                # Use the helper method to get smoothed values
                y_pred_smooth, _, pdf_smooth = self._smooth_quantile_function(
                    self.quantiles, y_pred_sample, num_points=num_points
                )
                
                results.append((y_pred_smooth, pdf_smooth))
        
        return results
    
    def save(self, filepath, format='cbm', compression_level=6):
        """
        Save the trained ProbGBT model to a file.
        
        For the MultiQuantile approach (train_separate_models=False), 
        this uses CatBoost's native save_model method.
        
        For the separate models approach (train_separate_models=True),
        this saves each model separately and compresses them into a tar.xz file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model. If using separate models, this should be a
            path ending with .tar.xz.
        format : str, default='cbm'
            Format to use when saving individual models. Options are:
            'cbm' - CatBoost binary format
            'json' - JSON format
            For the separate models approach, this specifies the format of 
            individual models within the archive.
        compression_level : int, default=6
            Compression level for xz compression (1-9, where 9 is highest).
            Only used when train_separate_models=True.
            
        Returns:
        --------
        None
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if not self.train_separate_models:
            # Save single model using CatBoost's native method
            self.model.save_model(filepath, format=format)
            
            # Save additional metadata
            metadata_path = f"{filepath}.metadata.json"
            metadata = {
                'num_quantiles': self.num_quantiles,
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'subsample': self.subsample,
                'random_seed': self.random_seed,
                'train_separate_models': self.train_separate_models,
                'quantiles': self.quantiles.tolist()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        else:
            # Save separate models to a tar.xz file
            # First, create a temporary directory to store models
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model metadata
                metadata = {
                    'num_quantiles': self.num_quantiles,
                    'iterations': self.iterations,
                    'learning_rate': self.learning_rate,
                    'depth': self.depth,
                    'subsample': self.subsample,
                    'random_seed': self.random_seed,
                    'train_separate_models': self.train_separate_models,
                    'quantiles': self.quantiles.tolist()
                }
                
                with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
                
                # Save each model separately
                model_dir = os.path.join(temp_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                
                # Create a mapping of quantile index to actual quantile value
                quantile_mapping = {i: q for i, q in enumerate(self.quantiles)}
                with open(os.path.join(temp_dir, 'quantile_mapping.json'), 'w') as f:
                    json.dump(quantile_mapping, f)
                
                # Use tqdm to show progress for saving individual models
                print("Saving individual models...")
                for i, q in enumerate(tqdm(self.quantiles)):
                    # Use index-based filenames instead of floating-point values
                    model_path = os.path.join(model_dir, f"quantile_{i:04d}.{format}")
                    self.trained_models[q].save_model(model_path, format=format)
                
                # Create tar.xz file with progress bar
                print(f"Compressing models to {filepath} (compression level: {compression_level})...")
                
                # Calculate total size for progress bar
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(temp_dir)
                    for filename in filenames
                )
                
                # Create a progress bar for compression
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Compressing") as pbar:
                    with tarfile.open(filepath, mode=f'w:xz', preset=compression_level) as tar:
                        for dirpath, _, filenames in os.walk(temp_dir):
                            for filename in filenames:
                                file_path = os.path.join(dirpath, filename)
                                arcname = os.path.relpath(file_path, temp_dir)
                                file_size = os.path.getsize(file_path)
                                tar.add(file_path, arcname=arcname)
                                pbar.update(file_size)
    
    def load(self, filepath, format='cbm'):
        """
        Load a previously saved ProbGBT model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file.
        format : str, default='cbm'
            Format of the saved model. Only used for single model approach.
            Options are:
            'cbm' - CatBoost binary format
            'json' - JSON format
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Check if the file is a tar.xz archive (separate models approach)
        if filepath.endswith('.tar.xz'):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the archive
                with tarfile.open(filepath, 'r:xz') as tar:
                    tar.extractall(path=temp_dir)
                
                # Load metadata
                with open(os.path.join(temp_dir, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                
                # Set model parameters from metadata
                self.num_quantiles = metadata['num_quantiles']
                self.iterations = metadata['iterations']
                self.learning_rate = metadata['learning_rate']
                self.depth = metadata['depth']
                self.subsample = metadata['subsample']
                self.random_seed = metadata['random_seed']
                self.train_separate_models = metadata['train_separate_models']
                self.quantiles = np.array(metadata['quantiles'])
                
                # Load quantile mapping
                with open(os.path.join(temp_dir, 'quantile_mapping.json'), 'r') as f:
                    quantile_mapping = json.load(f)
                
                # Convert string keys to integers (JSON serializes dict keys as strings)
                quantile_mapping = {int(k): float(v) for k, v in quantile_mapping.items()}
                
                # Load individual models
                model_dir = os.path.join(temp_dir, 'models')
                self.trained_models = {}
                
                print("Loading individual models...")
                # Process files in sorted order to ensure consistent loading
                model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('quantile_')])
                
                for filename in tqdm(model_files):
                    # Extract index from filename
                    idx = int(filename.split('_')[1].split('.')[0])
                    model_path = os.path.join(model_dir, filename)
                    
                    # Get the corresponding quantile value from the mapping
                    q = quantile_mapping[idx]
                    
                    model = CatBoostRegressor()
                    model.load_model(model_path, format=format)
                    
                    self.trained_models[q] = model
                
                # Ensure quantiles are in the correct order
                self.quantiles = np.array(sorted(self.trained_models.keys()))
        else:
            # Load single model
            self.model = CatBoostRegressor()
            self.model.load_model(filepath, format=format)
            
            # Load metadata if it exists
            metadata_path = f"{filepath}.metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Set model parameters from metadata
                self.num_quantiles = metadata['num_quantiles']
                self.iterations = metadata['iterations']
                self.learning_rate = metadata['learning_rate']
                self.depth = metadata['depth']
                self.subsample = metadata['subsample']
                self.random_seed = metadata['random_seed']
                self.train_separate_models = metadata['train_separate_models']
                self.quantiles = np.array(metadata['quantiles'])
            else:
                # If metadata file doesn't exist, infer quantiles from model
                self.train_separate_models = False
                
                # Set default parameters
                self.num_quantiles = 0  # Will be updated based on model
                self.iterations = 0  # Unknown
                self.learning_rate = None
                self.depth = None
                self.subsample = 1.0
                self.random_seed = 42
                
                # Try to extract quantiles from model
                try:
                    # Get loss function string from model
                    loss_function = self.model.get_param('loss_function')
                    
                    # Extract quantiles if it's a MultiQuantile loss function
                    if loss_function.startswith('MultiQuantile:alpha='):
                        alpha_str = loss_function.replace('MultiQuantile:alpha=', '')
                        self.quantiles = np.array([float(a.strip()) for a in alpha_str.split(',')])
                        self.num_quantiles = len(self.quantiles)
                    else:
                        # Default quantiles for backward compatibility
                        self.quantiles = np.linspace(0.01, 0.99, 50)
                        self.num_quantiles = 50
                except:
                    # Default quantiles for backward compatibility
                    self.quantiles = np.linspace(0.01, 0.99, 50)
                    self.num_quantiles = 50
        
        return self

    def evaluate_crps(self, X, y, subset_fraction=1.0, num_points=1000, verbose=True):
        """
        Evaluate the model using CRPS on test data.
        
        This method works for both individual and separate models.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Test features.
        y : numpy.ndarray
            Test target values.
        subset_fraction : float, default=1.0
            Fraction of test data to use for evaluation (0.0-1.0).
        num_points : int, default=1000
            Number of points to use for smoothing the PDF.
        verbose : bool, default=True
            If True, print progress information.
            
        Returns:
        --------
        float
            Mean CRPS value across test samples.
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Sample test data if subset_fraction < 1.0
        if 0.0 < subset_fraction < 1.0:
            test_size = int(len(X) * subset_fraction)
            idx = np.random.choice(len(X), test_size, replace=False)
            X_subset = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
            y_subset = y[idx]
        else:
            X_subset, y_subset = X, y
        
        if verbose:
            print(f"Evaluating CRPS on {len(X_subset)} test samples...")
        
        # Get quantile predictions
        y_pred = self.predict(X_subset, return_quantiles=True)
        
        # Calculate CRPS
        crps = self._compute_crps(y_subset, y_pred, num_points=num_points, verbose=verbose)
        
        if verbose:
            print(f"Test CRPS = {crps:.6f}")
        
        return crps 