import numpy as np
import pandas as pd
import os
import tarfile
import json
import tempfile
from catboost import CatBoostRegressor
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from pygam import LinearGAM, s
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class ProbGBT:
    """
    Probabilistic Gradient Boosted Trees model that provides uncertainty estimates.
    
    This model uses CatBoost's MultiQuantile loss function to predict multiple quantiles
    of the target distribution, then constructs a probability density function (PDF)
    from these quantiles.
    """
    
    def __init__(self, 
                 num_quantiles=100, 
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
        iterations : int, default=1000
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
    
    def train(self, X, y, cat_features=None, eval_set=None, use_best_model=True, verbose=True):
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
        use_best_model : bool, default=True
            If True, the best model on the validation set will be used.
        verbose : bool, default=True
            If True, print training progress.
            
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
                model.fit(X, y, eval_set=eval_set, use_best_model=use_best_model, verbose=False)
                
                # Add the model to the trained_models dictionary
                self.trained_models[q] = model
        else:
            # Format quantiles for CatBoost
            multiquantile_loss_str = str(self.quantiles.astype(float)).replace("[", "").replace("]", "").replace("\n", "").replace(" ", ", ")
            
            # Initialize CatBoost model
            self.model = CatBoostRegressor(
                cat_features=cat_features,
                loss_function=f"MultiQuantile:alpha={multiquantile_loss_str}",
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                subsample=self.subsample,
                random_seed=self.random_seed
            )
            
            # Train the model
            self.model.fit(X, y, eval_set=eval_set, use_best_model=use_best_model, verbose=verbose)
        
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
            # Add tqdm progress bar
            for i in tqdm(range(len(X)), desc="Making predictions"):
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
    
    def _get_smoothed_pdf(self, quantile_preds, i, num_points=1000, method='spline'):
        """
        Helper method to compute smoothed PDF for a single sample.
        
        Parameters:
        -----------
        quantile_preds : numpy.ndarray
            Quantile predictions from predict() with return_quantiles=True.
        i : int
            Index of the sample in quantile_preds to process.
        num_points : int, default=1000
            Number of points to use for the PDF.
        method : str, default='gmm'
            Method to use for PDF smoothing ('gmm' or 'spline').
            
        Returns:
        --------
        tuple: (x_values, pdf_values, cdf_values, quantiles_smooth)
            - x_values: x-axis values for the PDF (output range)
            - pdf_values: probability density function values
            - cdf_values: cumulative distribution function values 
            - quantiles_smooth: quantile values (0-1 range)
        """
        # Get the predicted quantiles for this sample
        y_pred_sample = quantile_preds[i]
        
        if method == 'spline':
            # Original method using GAM smoothing
            # Fit a GAM to smooth the quantile function
            gam = LinearGAM(s(0, constraints="monotonic_inc")).fit(self.quantiles, y_pred_sample)
            
            # Generate smoothed CDF
            quantiles_smooth = np.linspace(0, 1, num_points)
            y_pred_smooth = gam.predict(quantiles_smooth)
            
            # Compute PDF (derivative of the quantile function)
            epsilon = 1e-10  # Small value to avoid division by zero
            pdf_smooth = np.gradient(quantiles_smooth, y_pred_smooth + epsilon)
            
            # Simple method to ensure non-negativity
            pdf_smooth = np.maximum(pdf_smooth, 0)  # Ensure non-negative
            
            # Normalize the PDF
            integral = np.trapz(pdf_smooth, y_pred_smooth)
            if integral > 1e-10:  # Only normalize if integral is not too close to zero
                pdf_smooth /= integral
            else:
                # If integral is too small, use a uniform distribution instead
                pdf_smooth = np.ones_like(pdf_smooth) / len(pdf_smooth)
            
            # Return the quantiles_smooth as the cdf_values for consistency
            return y_pred_smooth, pdf_smooth, quantiles_smooth, quantiles_smooth
            
        elif method == 'gmm':
            # First get spline results as a base
            # Fit a GAM to smooth the quantile function
            gam = LinearGAM(s(0, constraints="monotonic_inc")).fit(self.quantiles, y_pred_sample)
            
            # Generate smoothed CDF
            quantiles_smooth = np.linspace(0, 1, num_points)
            y_pred_smooth = gam.predict(quantiles_smooth)
            
            # Compute PDF (derivative of the quantile function)
            epsilon = 1e-10  # Small value to avoid division by zero
            pdf_smooth = np.gradient(quantiles_smooth, y_pred_smooth + epsilon)
            
            # Simple method to ensure non-negativity
            pdf_smooth = np.maximum(pdf_smooth, 0)  # Ensure non-negative
            
            # Normalize the PDF
            integral = np.trapz(pdf_smooth, y_pred_smooth)
            if integral > 1e-10:
                pdf_smooth /= integral
            else:
                pdf_smooth = np.ones_like(pdf_smooth) / len(pdf_smooth)
            
            # Now apply GMM on top of spline results
            # Check for NaN or Inf values before proceeding with GMM
            if np.any(np.isnan(pdf_smooth)) or np.any(np.isinf(pdf_smooth)) or np.any(np.isnan(y_pred_smooth)) or np.any(np.isinf(y_pred_smooth)):
                # If there are NaNs or Infs, skip GMM and use spline results directly
                x_values = y_pred_smooth
                pdf_values = pdf_smooth
            else:
                try:
                    # Use a gaussian mixture model
                    gmm = GaussianMixture(n_components=3, max_iter=1000)
                    gmm.fit(y_pred_smooth.reshape(-1, 1))
                    
                    # Check if GMM converged
                    if not gmm.converged_:
                        print(f"Warning: GMM did not converge after {gmm.n_iter_} iterations, using spline result")
                        x_values = y_pred_smooth
                        pdf_values = pdf_smooth
                    else:
                        # Generate final smoothed PDF on a regular grid
                        x_values = np.linspace(np.min(y_pred_smooth), np.max(y_pred_smooth), num_points)
                        pdf_values = np.exp(gmm.score_samples(x_values.reshape(-1, 1)))
                        
                        # Much more aggressive checks for problematic GMM outputs
                        use_gmm = True
                        
                        # 1. Check for any invalid values
                        if np.any(np.isnan(pdf_values)) or np.any(np.isinf(pdf_values)) or np.all(pdf_values < 1e-10):
                            print(f"Warning: GMM produced invalid values, using spline result")
                            use_gmm = False
                            
                        # 2. Check for extremely peaked distributions (very common failure mode)
                        if use_gmm:
                            pdf_max = np.max(pdf_values)
                            pdf_mean = np.mean(pdf_values)
                            if pdf_max > 100 * pdf_mean:  # Less extreme threshold to be more cautious
                                print(f"Warning: GMM produced extremely peaked distribution (max/mean ratio: {pdf_max/pdf_mean:.1f}), using spline result")
                                use_gmm = False
                        
                        # 3. Check for too narrow effective support (another common failure)
                        if use_gmm:
                            # Count points with significant probability mass
                            significant_points = np.sum(pdf_values > pdf_max * 0.01)
                            if significant_points < num_points * 0.01:  # Less than 1% of points have significant probability
                                print(f"Warning: GMM output has too narrow effective support ({significant_points}/{num_points} significant points), using spline result")
                                use_gmm = False
                        
                        # 4. Generate a test CDF and check its properties
                        if use_gmm:
                            # Normalize PDF first
                            integral = np.trapz(pdf_values, x_values)
                            if integral <= 1e-10:
                                print(f"Warning: GMM produced too small integral ({integral}), using spline result")
                                use_gmm = False
                            else:
                                # Normalize and compute test CDF
                                pdf_values /= integral
                                test_cdf = cumulative_trapezoid(pdf_values, x_values, initial=0)
                                
                                # Check that CDF is usable for confidence intervals
                                if test_cdf[-1] <= 0.9:  # CDF should end close to 1
                                    print(f"Warning: GMM produced CDF that doesn't reach 1 (max: {test_cdf[-1]:.3f}), using spline result")
                                    use_gmm = False
                                    
                                # Check if CDF has enough distinct steps for quantile calculation
                                elif np.count_nonzero(np.diff(test_cdf) > 1e-6) < 20:
                                    print(f"Warning: GMM produced CDF with too few increasing steps, using spline result")
                                    use_gmm = False
                                    
                                # Check if the CDF can extract reasonable quantiles
                                elif any(np.isclose(np.searchsorted(test_cdf, q) / len(test_cdf), 0) or 
                                         np.isclose(np.searchsorted(test_cdf, q) / len(test_cdf), 1) 
                                         for q in [0.025, 0.25, 0.5, 0.75, 0.975]):
                                    print(f"Warning: GMM produced CDF that would give extreme quantiles, using spline result")
                                    use_gmm = False
                        
                        # Finally, use spline results if any test failed
                        if not use_gmm:
                            x_values = y_pred_smooth
                            pdf_values = pdf_smooth
                        
                except Exception as e:
                    # If GMM fails, fall back to the spline result
                    print(f"Warning: GMM failed, using spline result: {e}")
                    x_values = y_pred_smooth
                    pdf_values = pdf_smooth
            
            # Compute CDF using cumulative_trapezoid for accurate integration
            cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
            
            # Normalize CDF to ensure it ends at 1.0
            if cdf_values[-1] > 0:
                cdf_values /= cdf_values[-1]
            else:
                print(f"Warning: CDF ends at zero, falling back to uniform CDF")
                cdf_values = np.linspace(0, 1, len(x_values))
                
            # Ensure CDF is strictly increasing (important for accurate quantile lookup)
            # Find places where CDF doesn't increase
            not_increasing = np.where(np.diff(cdf_values) <= 0)[0]
            if len(not_increasing) > 0:
                # Apply a small correction where needed
                epsilon = 1e-10
                for idx in not_increasing:
                    cdf_values[idx+1] = cdf_values[idx] + epsilon
                
                # Re-normalize to ensure CDF ends at 1.0
                cdf_values /= cdf_values[-1]
                
            return x_values, pdf_values, cdf_values, quantiles_smooth
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'spline' or 'gmm'.")

    def predict_interval(self, X, confidence_level=0.95, method='spline', num_points=1000):
        """
        Predict confidence intervals for the given samples using the smoothed PDF.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        confidence_level : float, default=0.95
            Confidence level for the interval (between 0 and 1).
        method : str, default='gmm'
            Method to use for PDF smoothing ('gmm' or 'spline').
        num_points : int, default=1000
            Number of points to use for the PDF.
            
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
        
        # Add tqdm progress bar
        for i in tqdm(range(quantile_preds.shape[0]), desc="Processing samples"):
            # Get smoothed PDF and CDF (only compute once)
            x_values, pdf_values, cdf_values, quantiles_smooth = self._get_smoothed_pdf(
                quantile_preds, i, num_points=num_points, method=method
            )
            
            # Compute the quantile values corresponding to the confidence interval
            lower_q = (1 - confidence_level) / 2
            upper_q = 1 - (1 - confidence_level) / 2
            
            # Using 'right' side to find the correct index for lower bound
            # (the first index where CDF value is >= lower_quantile)
            lower_idx = np.searchsorted(cdf_values, lower_q, side='right')
            
            # Using 'left' side to find the correct index for upper bound
            # (the first index where CDF value is >= upper_quantile)
            upper_idx = np.searchsorted(cdf_values, upper_q, side='left')
            
            # Ensure indices are within bounds
            lower_idx = max(0, min(lower_idx, len(x_values) - 1))
            upper_idx = max(0, min(upper_idx, len(x_values) - 1))
            
            # Get the x values at those indices
            lower_bound = x_values[lower_idx]
            upper_bound = x_values[upper_idx]
            
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def predict_pdf(self, X, num_points=1000, method='spline'):
        """
        Predict the probability density function for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        num_points : int, default=1000
            Number of points to use for the PDF.
        method : str, default='spline'
            Method to use for PDF smoothing:
            - 'spline': Original method using GAM smoothing only (recommended)
            - 'gmm': Kernel Density Estimation applied on top of spline smoothing 
            
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
        
        # Add tqdm progress bar
        for i in tqdm(range(quantile_preds.shape[0]), desc="Generating PDF"):
            # Use the helper method to get smoothed PDF
            x_values, pdf_values, _, _ = self._get_smoothed_pdf(
                quantile_preds, i, num_points=num_points, method=method
            )
            
            results.append((x_values, pdf_values))
        
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
                    'quantiles': self.quantiles.tolist(),
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
        Load a saved ProbGBT model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file.
        format : str, default='cbm'
            Format of the saved model. Options are:
            'cbm' - CatBoost binary format
            'json' - JSON format
            This is only used for loading individual models if loading from a tar.xz archive.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        if filepath.endswith('.tar.xz'):
            # Load from tar.xz (separate models approach)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the tar.xz file with progress bar
                file_size = os.path.getsize(filepath)
                
                print(f"Extracting archive {filepath}...")
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    with tarfile.open(filepath, mode='r:xz') as tar:
                        members = tar.getmembers()
                        for member in members:
                            tar.extract(member, path=temp_dir)
                            pbar.update(file_size // len(members))  # Approximate progress
                
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
            # Load single model using CatBoost's native method
            self.train_separate_models = False
            self.model = CatBoostRegressor()
            self.model.load_model(filepath, format=format)
            
            # Try to extract quantiles from the model parameters
            loss_function = self.model.get_param('loss_function')
            if loss_function and 'MultiQuantile:alpha=' in loss_function:
                quantiles_str = loss_function.split('MultiQuantile:alpha=')[1]
                try:
                    self.quantiles = np.array([float(q.strip()) for q in quantiles_str.split(',')])
                except:
                    # If we can't parse the quantiles, create default ones
                    self.quantiles = self._generate_non_uniform_quantiles()
            else:
                # If we can't extract quantiles from the model, generate default ones
                self.quantiles = self._generate_non_uniform_quantiles()
        
        return self 