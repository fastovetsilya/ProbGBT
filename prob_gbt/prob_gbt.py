import numpy as np
import pandas as pd
import os
import tarfile
import json
import tempfile
import shutil
from catboost import CatBoostRegressor
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from pygam import LinearGAM, s
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

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
        
        for i in range(quantile_preds.shape[0]):
            # Get the predicted quantiles for this sample
            y_pred_sample = quantile_preds[i]
            
            # Fit a GAM to smooth the quantile function
            gam = LinearGAM(s(0, constraints="monotonic_inc")).fit(self.quantiles, y_pred_sample)
            
            # Generate smoothed CDF
            quantiles_smooth = np.linspace(0, 1, 1000)
            y_pred_smooth = gam.predict(quantiles_smooth)
            
            # Find the quantile values corresponding to the confidence interval
            lower_idx = np.searchsorted(quantiles_smooth, (1 - confidence_level) / 2)
            upper_idx = np.searchsorted(quantiles_smooth, 1 - (1 - confidence_level) / 2)
            
            lower_bounds.append(y_pred_smooth[lower_idx])
            upper_bounds.append(y_pred_smooth[upper_idx])
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def predict_pdf(self, X, num_points=1000, method='kde', gaussian_sigma=1.0):
        """
        Predict the probability density function for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        num_points : int, default=1000
            Number of points to use for the PDF.
        method : str, default='kde'
            Method to use for PDF smoothing:
            - 'kde': Kernel Density Estimation with isotonic regression (recommended)
            - 'spline': Original method using GAM smoothing
        gaussian_sigma : float, default=1.0
            Sigma parameter for Gaussian filter smoothing if used instead of isotonic regression.
            Higher values create smoother PDFs.
            
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
        
        if method == 'spline':
            # Original method using GAM smoothing
            for i in range(quantile_preds.shape[0]):
                # Get the predicted quantiles for this sample
                y_pred_sample = quantile_preds[i]
                
                # Fit a GAM to smooth the quantile function
                gam = LinearGAM(s(0, constraints="monotonic_inc")).fit(self.quantiles, y_pred_sample)
                
                # Generate smoothed CDF
                quantiles_smooth = np.linspace(0, 1, num_points)
                y_pred_smooth = gam.predict(quantiles_smooth)
                
                # Compute PDF (derivative of the quantile function)
                epsilon = 1e-10  # Small value to avoid division by zero
                pdf_smooth = np.gradient(quantiles_smooth, y_pred_smooth + epsilon)
                pdf_smooth = np.maximum(pdf_smooth, 0)  # Ensure non-negative
                
                # Normalize the PDF
                pdf_smooth /= np.trapz(pdf_smooth, y_pred_smooth)
                
                results.append((y_pred_smooth, pdf_smooth))
        
        elif method == 'kde':
            # Method using spline smoothing, isotonic regression, and KDE
            for i in range(quantile_preds.shape[0]):
                # Get the predicted quantiles for this sample
                y_pred_sample = quantile_preds[i]
                
                # Step 1: Generate smoothed CDF based on quantiles
                # Fit a GAM to smooth the quantile function
                gam = LinearGAM(s(0, constraints="monotonic_inc")).fit(self.quantiles, y_pred_sample)
                gam.gridsearch(self.quantiles, y_pred_sample)

                # Generate smoothed CDF
                quantiles_smooth = np.linspace(0, 1, num_points)
                y_cdf = gam.predict(quantiles_smooth)
                
                # Step 2: Compute the naive PDF using finite differences
                dx = np.diff(y_cdf)
                dy = np.diff(quantiles_smooth)
                pdf_naive = dy / (dx + 1e-10)  # Small epsilon to avoid division by zero
                x_pdf = y_cdf[:-1] + dx/2  # Centers of the intervals
                
                # Step 3: Apply isotonic regression to ensure non-negativity
                iso_reg = IsotonicRegression(y_min=0)
                pdf_iso = iso_reg.fit_transform(x_pdf, pdf_naive)
                
                # Step 4: Normalize the PDF
                # Add safeguard for division by zero
                integral = np.trapz(pdf_iso, x_pdf)
                if integral > 1e-10:  # Only normalize if integral is not too close to zero
                    pdf_iso /= integral
                else:
                    # If integral is too small, use a uniform distribution instead
                    pdf_iso = np.ones_like(pdf_iso) / len(pdf_iso)
                
                # Step 5: Apply KDE for further smoothing with cross-validation
                # Find optimal bandwidth using cross-validation
                best_score = float('-inf')
                best_bw = 0.01  # Default value
                
                # Check for NaN or Inf values before proceeding
                if np.any(np.isnan(pdf_iso)) or np.any(np.isinf(pdf_iso)) or np.any(np.isnan(x_pdf)) or np.any(np.isinf(x_pdf)):
                    # If there are NaNs or Infs, skip KDE and use the normalized PDF directly
                    x_kde = x_pdf
                    pdf_kde = pdf_iso
                else:
                    # Cross-validate with different bandwidths if we have enough data points
                    if len(x_pdf) > 10:
                        bandwidths = np.logspace(-3, 0, 10)  # Try different bandwidths
                        kf = KFold(n_splits=min(5, len(x_pdf)), shuffle=True, random_state=self.random_seed)
                        
                        for bw in bandwidths:
                            cv_scores = []
                            for train_idx, test_idx in kf.split(x_pdf):
                                if len(train_idx) < 3:  # Skip if not enough training data
                                    continue
                                
                                # Make sure there are no NaN or Inf values
                                if np.any(np.isnan(pdf_iso[train_idx])) or np.any(np.isinf(pdf_iso[train_idx])):
                                    continue
                                
                                try:
                                    # Train KDE on training indices
                                    kde = gaussian_kde(x_pdf[train_idx], weights=pdf_iso[train_idx], bw_method=bw)
                                    
                                    # Evaluate on test indices
                                    score = np.mean(kde.logpdf(x_pdf[test_idx]))
                                    cv_scores.append(score)
                                except Exception as e:
                                    # If there's an error in KDE, skip this bandwidth
                                    print(f"Warning: KDE error with bandwidth {bw}: {e}")
                                    continue
                            
                            if cv_scores and np.mean(cv_scores) > best_score:
                                best_score = np.mean(cv_scores)
                                best_bw = bw
                    
                    try:
                        # Apply KDE with best bandwidth
                        kde = gaussian_kde(x_pdf, weights=pdf_iso, bw_method=best_bw)
                        
                        # Generate final smoothed PDF on a regular grid
                        x_kde = np.linspace(np.min(x_pdf), np.max(x_pdf), num_points)
                        pdf_kde = kde(x_kde)
                        
                        # Final normalization
                        integral = np.trapz(pdf_kde, x_kde)
                        if integral > 1e-10:
                            pdf_kde /= integral
                        else:
                            # If integral is too small, use a uniform distribution
                            pdf_kde = np.ones_like(x_kde) / len(x_kde)
                            
                    except Exception as e:
                        # If KDE fails, fall back to the normalized isotonic regression result
                        print(f"Warning: KDE failed, using isotonic regression result: {e}")
                        x_kde = x_pdf
                        pdf_kde = pdf_iso
                
                results.append((x_kde, pdf_kde))
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'spline' or 'kde'.")
        
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