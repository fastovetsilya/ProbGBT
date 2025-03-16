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
from sklearn.neighbors import KernelDensity
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
                 train_separate_models=False,
                 calibrate=True):
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
        calibrate : bool, default=False
            If True, use conformal prediction to calibrate predicted quantiles.
        """
        self.num_quantiles = num_quantiles
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.subsample = subsample
        self.random_seed = random_seed
        self.train_separate_models = train_separate_models
        self.calibrate = calibrate
        self.model = None
        self.trained_models = {}
        self.quantiles = None
        self.conformity_scores = None
        self.is_calibrated = False
        self._last_pdfs = None
        
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
    
    def train(self, X, y, cat_features=None, eval_set=None, use_best_model=True, verbose=True,
             calibration_set=None, calibration_size=0.2):
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
        calibration_set : tuple, optional
            (X_cal, y_cal) for conformal calibration. If provided, this will be used instead of 
            splitting from training data. Only used when calibrate=True.
        calibration_size : float, default=0.2
            Fraction of training data to use for calibration if calibration_set is not provided.
            Only used when calibrate=True.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Generate quantiles
        self.quantiles = self._generate_non_uniform_quantiles()
        
        # If calibration is enabled and no calibration set is provided, split the training data
        X_train, y_train = X, y
        X_cal, y_cal = None, None
        
        if self.calibrate and calibration_set is None:
            # Create a calibration set by splitting the training data
            if verbose:
                print(f"Splitting {calibration_size:.0%} of training data for calibration")
            
            # Random shuffling of indices
            n_samples = len(y)
            indices = np.arange(n_samples)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            
            # Determine split point
            n_calibration = int(n_samples * calibration_size)
            
            # Split data
            train_indices = indices[n_calibration:]
            cal_indices = indices[:n_calibration]
            
            X_train = X.iloc[train_indices] if isinstance(X, pd.DataFrame) else X[train_indices]
            y_train = y[train_indices]
            X_cal = X.iloc[cal_indices] if isinstance(X, pd.DataFrame) else X[cal_indices]
            y_cal = y[cal_indices]
        elif self.calibrate and calibration_set is not None:
            # Use the provided calibration set
            X_cal, y_cal = calibration_set
            if verbose:
                print(f"Using provided calibration set with {len(y_cal)} samples")
        
        # Train on the training set (or full set if not calibrating)
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
                model.fit(X_train, y_train, eval_set=eval_set, use_best_model=use_best_model, verbose=False)
                
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
            self.model.fit(X_train, y_train, eval_set=eval_set, use_best_model=use_best_model, verbose=verbose)
        
        # Perform calibration if enabled
        if self.calibrate and X_cal is not None and y_cal is not None:
            self._calibrate(X_cal, y_cal, verbose)
        
        return self
    
    def _calibrate(self, X_cal, y_cal, verbose=True):
        """
        Calibrate the model using conformal prediction.
        
        Parameters:
        -----------
        X_cal : pandas.DataFrame or numpy.ndarray
            Calibration features.
        y_cal : numpy.ndarray
            Calibration target values.
        verbose : bool, default=True
            If True, print calibration progress.
        
        Returns:
        --------
        None
        """
        if verbose:
            print("Calibrating model using conformal prediction...")
        
        # Get uncalibrated quantile predictions on calibration set
        quantile_preds = self.predict(X_cal, return_quantiles=True)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        # Calculate nonconformity scores for each quantile
        n_samples = len(y_cal)
        n_quantiles = len(self.quantiles)
        
        # Store calibration information
        self.conformity_scores = {}
        
        # For each quantile, compute nonconformity scores
        for q_idx, q in enumerate(self.quantiles):
            # Extract predictions for this quantile
            q_preds = quantile_preds[:, q_idx]
            
            # For lower tail: E_i = y_i - q̂_α(X_i)
            # We want to adjust the quantile to ensure P(Y ≤ q̂_α(X)) = α
            conformity_scores = y_cal - q_preds
            
            # Store the scores for this quantile
            self.conformity_scores[q] = conformity_scores
        
        # Verify basic properties if verbose is enabled
        if verbose:
            for q in [0.1, 0.5, 0.9]:
                if q in self.quantiles:
                    q_idx = np.where(self.quantiles == q)[0][0]
                    # Get predictions for this quantile
                    q_preds = quantile_preds[:, q_idx]
                    # Calculate empirical coverage (proportion of y values below prediction)
                    empirical_coverage = np.mean(y_cal <= q_preds)
                    print(f"Quantile {q}: Desired coverage = {q*100:.1f}%, Raw empirical coverage = {empirical_coverage*100:.1f}%")
        
        self.is_calibrated = True
        if verbose:
            print("Calibration completed successfully.")
    
    def predict(self, X, return_quantiles=False, method='sample_kde', num_points=1000):
        """
        Predict the target values for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        return_quantiles : bool, default=False
            Whether to return the predictions for all quantiles.
        method : str, default='sample_kde'
            Method to use for PDF smoothing:
            - 'spline': Original method using GAM smoothing only
            - 'gmm': Kernel Density Estimation applied on top of spline smoothing
            - 'sample_kde': Sample from empirical CDF and apply KDE smoothing (default)
        num_points : int, default=1000
            Number of points to use for the PDF.
            
        Returns:
        --------
        numpy.ndarray: Predictions for the samples.
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get raw, uncalibrated quantile predictions from the model
        quantile_preds = self._get_uncalibrated_predictions(X)
        
        # Apply calibration if enabled and model is calibrated
        if self.calibrate and self.is_calibrated:
            # Make a copy of the predictions to avoid modifying the original
            calibrated_preds = quantile_preds.copy()
            
            # For each quantile, apply conformal calibration
            for q_idx, q in enumerate(self.quantiles):
                scores = self.conformity_scores[q]
                n_cal = len(scores)
                
                # Calculate the adjustment using proper conformal prediction
                # For each quantile q, we find s_hat such that:
                # P(Y - q̂_α(X) ≤ s_hat) = q
                # This is the (n_cal+1)*q / n_cal quantile of the scores
                
                # Handle extreme quantiles carefully
                if q < 0.01:
                    # For very low quantiles, be conservative
                    s_hat = np.min(scores)
                elif q > 0.99:
                    # For very high quantiles, be conservative
                    s_hat = np.max(scores)
                else:
                    # Find the empirical quantile of the scores
                    # The (n+1)q/n formula is a finite-sample correction
                    quantile_level = min(1.0, (n_cal + 1) * q / n_cal)
                    s_hat = np.quantile(scores, quantile_level)
                
                # The calibrated prediction is q̂_α(X) + s_hat
                calibrated_preds[:, q_idx] = quantile_preds[:, q_idx] + s_hat
            
            # Replace the raw predictions with calibrated predictions
            quantile_preds = calibrated_preds
        
        if return_quantiles:
            return quantile_preds
        
        # Calculate median predictions using the specified smoothing method
        medians = []
        
        for i in tqdm(range(quantile_preds.shape[0]), desc="Calculating medians"):
            # Get smoothed PDF and CDF using the specified method
            x_values, pdf_values, cdf_values, _ = self._get_smoothed_pdf(
                quantile_preds, i, num_points=num_points, method=method
            )
            
            # Find the median (point where CDF crosses 0.5)
            median_idx = np.searchsorted(cdf_values, 0.5, side='left')
            median_idx = max(0, min(median_idx, len(x_values) - 1))
            median = x_values[median_idx]
            medians.append(median)
        
        return np.array(medians)
    
    def _get_smoothed_pdf(self, quantile_preds, i, num_points=1000, method='sample_kde'):
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
        method : str, default='sample_kde'
            Method to use for PDF smoothing:
            - 'spline': Original method using GAM smoothing only
            - 'gmm': Kernel Density Estimation applied on top of spline smoothing
            - 'sample_kde': Sample from empirical CDF and apply KDE smoothing (default)
            
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
        
        if method == 'sample_kde':
            # Determine the range directly from the quantile predictions
            y_min, y_max = np.min(y_pred_sample), np.max(y_pred_sample)
            y_range = y_max - y_min
            
            # Add padding to the range to avoid edge effects
            y_min -= 0.2 * y_range
            y_max += 0.2 * y_range
            
            # Generate random uniform samples for probability values
            np.random.seed(self.random_seed)  # For reproducibility
            n_samples = num_points  # Number of samples to draw (same as num_points)
            prob_samples = np.random.uniform(0, 1, n_samples)
            
            # Interpolate to get samples from the empirical CDF
            samples = np.interp(prob_samples, self.quantiles, y_pred_sample)
            
            # Reshape samples for KDE
            samples = samples.reshape(-1, 1)
            
            # Fit KDE and get smoothed PDF
            kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            kde.fit(samples)
            
            # Generate points for PDF evaluation
            x_values = np.linspace(y_min, y_max, num_points).reshape(-1, 1)
            log_pdf = kde.score_samples(x_values)
            pdf_values = np.exp(log_pdf)
            
            # Normalize PDF
            pdf_values = pdf_values / np.trapz(pdf_values, x_values.ravel())
            
            # Compute CDF using numerical integration
            cdf_values = cumulative_trapezoid(pdf_values, x_values.ravel(), initial=0)
            
            # Ensure CDF ends at 1.0
            cdf_values /= cdf_values[-1]
            
            # Create quantiles_smooth for API consistency
            quantiles_smooth = np.linspace(0, 1, num_points)
            
            return x_values.ravel(), pdf_values, cdf_values, quantiles_smooth
        
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
            raise ValueError(f"Unknown method: {method}. Choose from 'spline', 'gmm', or 'sample_kde'.")

    def predict_interval(self, X, confidence_level=0.95, method='sample_kde', num_points=1000):
        """
        Predict confidence intervals for the given samples using the smoothed PDF.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        confidence_level : float, default=0.95
            Confidence level for the interval (between 0 and 1).
        method : str, default='sample_kde'
            Method to use for PDF smoothing ('gmm', 'spline', or 'sample_kde').
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
        
        # Get quantile predictions (these will already be calibrated if calibration is enabled)
        quantile_preds = self.predict(X, return_quantiles=True)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        # Calculate the lower and upper quantiles for the confidence interval
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q
        
        # Check if we should use the direct quantile approach for calibrated models
        # This is more reliable for ensuring proper coverage
        if self.calibrate and self.is_calibrated:
            # Find the closest available quantiles in our model
            lower_idx = np.argmin(np.abs(self.quantiles - lower_q))
            upper_idx = np.argmin(np.abs(self.quantiles - upper_q))
            
            # If the closest lower quantile is larger than our target, move one index back if possible
            if self.quantiles[lower_idx] > lower_q and lower_idx > 0:
                lower_idx -= 1
                
            # If the closest upper quantile is smaller than our target, move one index forward if possible
            if self.quantiles[upper_idx] < upper_q and upper_idx < len(self.quantiles) - 1:
                upper_idx += 1
                
            # Get the predicted values at those quantiles
            lower_bounds = quantile_preds[:, lower_idx]
            upper_bounds = quantile_preds[:, upper_idx]
            
            # Now get smoothed PDFs based on calibrated quantiles for better visualization
            # But use the direct quantile predictions for the intervals
            pdfs = []
            for i in tqdm(range(quantile_preds.shape[0]), desc="Processing samples"):
                # Get smoothed PDF (only compute for visualization - not for intervals)
                x_values, pdf_values, _, _ = self._get_smoothed_pdf(
                    quantile_preds, i, num_points=num_points, method=method
                )
                pdfs.append((x_values, pdf_values))
            
            # Store PDFs as a class attribute for later access
            self._last_pdfs = pdfs
            
            # Ensure lower bounds are always less than upper bounds
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            # Find cases where bounds are reversed
            swap_indices = np.where(lower_bounds > upper_bounds)[0]
            if len(swap_indices) > 0:
                # Swap the values where needed
                temp = lower_bounds[swap_indices].copy()
                lower_bounds[swap_indices] = upper_bounds[swap_indices]
                upper_bounds[swap_indices] = temp
                
                # Log warning if bounds were swapped
                if len(swap_indices) > 0:
                    print(f"Warning: Swapped {len(swap_indices)} lower/upper bounds where lower > upper.")
            
            return lower_bounds, upper_bounds
        
        # If not calibrated, use smoothed approach
        lower_bounds = []
        upper_bounds = []
        pdfs = []
        
        # Add tqdm progress bar
        for i in tqdm(range(quantile_preds.shape[0]), desc="Processing samples"):
            # Get smoothed PDF and CDF (only compute once)
            x_values, pdf_values, cdf_values, quantiles_smooth = self._get_smoothed_pdf(
                quantile_preds, i, num_points=num_points, method=method
            )
            
            # Store PDF for later reference
            pdfs.append((x_values, pdf_values))
            
            # For both bounds, use 'left' side for consistent quantile extraction
            # This finds the smallest x value such that CDF(x) >= target_probability
            lower_idx = np.searchsorted(cdf_values, lower_q, side='left')
            upper_idx = np.searchsorted(cdf_values, upper_q, side='left')
            
            # Ensure indices are within bounds
            lower_idx = max(0, min(lower_idx, len(x_values) - 1))
            upper_idx = max(0, min(upper_idx, len(x_values) - 1))
            
            # Get the x values at those indices
            lower_bound = x_values[lower_idx]
            upper_bound = x_values[upper_idx]
            
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        # Store PDFs as a class attribute for later access
        self._last_pdfs = pdfs
        
        # Ensure lower bounds are always less than upper bounds
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        
        # Find cases where bounds are reversed
        swap_indices = np.where(lower_bounds > upper_bounds)[0]
        if len(swap_indices) > 0:
            # Swap the values where needed
            temp = lower_bounds[swap_indices].copy()
            lower_bounds[swap_indices] = upper_bounds[swap_indices]
            upper_bounds[swap_indices] = temp
            
            # Log warning if bounds were swapped
            if len(swap_indices) > 0:
                print(f"Warning: Swapped {len(swap_indices)} lower/upper bounds where lower > upper.")
        
        return lower_bounds, upper_bounds
    
    def _get_uncalibrated_predictions(self, X):
        """
        Get raw, uncalibrated quantile predictions directly from the model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
            
        Returns:
        --------
        numpy.ndarray: Raw uncalibrated quantile predictions with shape (n_samples, n_quantiles)
        """
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
            for i in tqdm(range(len(X)), desc="Getting uncalibrated predictions"):
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
            
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
            
        return quantile_preds
        
    def predict_pdf(self, X, num_points=1000, method='sample_kde', use_calibration=True):
        """
        Predict the probability density function for the given samples.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        num_points : int, default=1000
            Number of points to use for the PDF.
        method : str, default='sample_kde'
            Method to use for PDF smoothing:
            - 'spline': Original method using GAM smoothing only
            - 'gmm': Kernel Density Estimation applied on top of spline smoothing
            - 'sample_kde': Sample from empirical CDF and apply KDE smoothing (default)
        use_calibration : bool, default=True
            Whether to base the PDF on calibrated quantiles (if available).
            - True: Uses calibrated quantiles for better statistical coverage
            - False: Uses raw uncalibrated quantiles, which may give smoother but less accurate distributions
            
        Returns:
        --------
        list of tuples: [(x_values, pdf_values), ...] for each sample
        """
        if self.train_separate_models and not self.trained_models:
            raise ValueError("Models have not been trained yet. Call train() first.")
        elif not self.train_separate_models and self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Determine whether to use calibrated predictions
        use_cal = use_calibration and self.calibrate and self.is_calibrated
        
        if use_cal:
            # Get calibrated quantile predictions
            quantile_preds = self.predict(X, return_quantiles=True)
        else:
            # Get uncalibrated quantile predictions directly from the model
            quantile_preds = self._get_uncalibrated_predictions(X)
        
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
            
            # If calibrated, save calibration information
            if self.calibrate and self.is_calibrated:
                # For single model, save calibration info separately
                calibration_path = f"{filepath}.calibration.json"
                calibration_data = {
                    'quantiles': self.quantiles.tolist(),
                    'conformity_scores': {str(q): scores.tolist() for q, scores in self.conformity_scores.items()},
                    'is_calibrated': self.is_calibrated,
                    'calibrate': self.calibrate
                }
                with open(calibration_path, 'w') as f:
                    json.dump(calibration_data, f)
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
                    'calibrate': self.calibrate,
                    'is_calibrated': self.is_calibrated
                }
                
                with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
                
                # Save calibration data if available
                if self.calibrate and self.is_calibrated:
                    calibration_data = {str(q): scores.tolist() for q, scores in self.conformity_scores.items()}
                    with open(os.path.join(temp_dir, 'calibration.json'), 'w') as f:
                        json.dump(calibration_data, f)
                
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
                self.calibrate = metadata.get('calibrate', False)
                self.is_calibrated = metadata.get('is_calibrated', False)
                
                # Load calibration data if available
                if self.calibrate and self.is_calibrated:
                    calibration_path = os.path.join(temp_dir, 'calibration.json')
                    if os.path.exists(calibration_path):
                        with open(calibration_path, 'r') as f:
                            calibration_data = json.load(f)
                            # Convert string keys back to float
                            self.conformity_scores = {float(q): np.array(scores) for q, scores in calibration_data.items()}
                    else:
                        # If calibration data is missing, set is_calibrated to False
                        self.is_calibrated = False
                
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
            
            # Reset calibration attributes
            self.calibrate = False
            self.is_calibrated = False
            self.conformity_scores = None
            
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
            
            # Check for calibration data
            calibration_path = f"{filepath}.calibration.json"
            if os.path.exists(calibration_path):
                try:
                    with open(calibration_path, 'r') as f:
                        calibration_data = json.load(f)
                        self.quantiles = np.array(calibration_data['quantiles'])
                        self.conformity_scores = {float(q): np.array(scores) for q, scores in calibration_data['conformity_scores'].items()}
                        self.is_calibrated = calibration_data['is_calibrated']
                        self.calibrate = calibration_data['calibrate']
                except Exception as e:
                    print(f"Warning: Failed to load calibration data: {e}")
        
        return self 

    def evaluate_calibration(self, X_val, y_val, confidence_levels=None):
        """
        Evaluate the calibration quality of the model on validation data.
        
        Parameters:
        -----------
        X_val : pandas.DataFrame or numpy.ndarray
            Validation features.
        y_val : numpy.ndarray
            Validation target values.
        confidence_levels : list, optional
            List of confidence levels to evaluate. If None, uses [0.5, 0.8, 0.9, 0.95, 0.99].
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing desired vs. actual coverage for each confidence level.
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        
        # Get predictions for all quantiles
        quantile_preds = self.predict(X_val, return_quantiles=True)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        # Calculate coverage for each confidence level
        coverage_results = []
        for conf_level in confidence_levels:
            # Calculate the lower and upper quantiles for this confidence level
            lower_q = (1 - conf_level) / 2
            upper_q = 1 - lower_q
            
            # Find indices for the closest quantiles
            lower_idx = np.argmin(np.abs(self.quantiles - lower_q))
            upper_idx = np.argmin(np.abs(self.quantiles - upper_q))
            
            # Extract predictions for these quantiles
            lower_bounds = quantile_preds[:, lower_idx]
            upper_bounds = quantile_preds[:, upper_idx]
            
            # Calculate coverage (percentage of true values within the bounds)
            coverage = np.mean((y_val >= lower_bounds) & (y_val <= upper_bounds))
            
            # Calculate average interval width
            avg_width = np.mean(upper_bounds - lower_bounds)
            
            coverage_results.append({
                'confidence_level': conf_level,
                'desired_coverage': conf_level,
                'actual_coverage': coverage,
                'average_width': avg_width,
                'coverage_error': coverage - conf_level,
                'lower_quantile': self.quantiles[lower_idx],
                'upper_quantile': self.quantiles[upper_idx]
            })
        
        # Return results as DataFrame
        return pd.DataFrame(coverage_results) 

    def predict_distribution(self, X, confidence_levels=None, method='sample_kde', num_points=1000):
        """
        Predict both calibrated confidence intervals and a smooth probability distribution.
        
        This method provides a complete probabilistic forecast including:
        1. Calibrated intervals with proper coverage guarantees
        2. Smooth probability distribution using the same calibrated quantiles
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to predict on.
        confidence_levels : list of float, optional
            List of confidence levels for prediction intervals. If None, uses [0.5, 0.8, 0.9, 0.95, 0.99].
        method : str, default='sample_kde'
            Method to use for PDF smoothing ('gmm', 'spline', or 'sample_kde').
        num_points : int, default=1000
            Number of points to use for the PDF.
            
        Returns:
        --------
        dict: A dictionary containing:
            - 'intervals': A dict mapping confidence levels to (lower_bounds, upper_bounds) tuples
            - 'pdf': A list of tuples (x_values, pdf_values) for each sample
            - 'mean': Mean prediction for each sample
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        
        results = {}
        
        # Get mean predictions
        results['mean'] = self.predict(X)
        
        # Get calibrated intervals for each confidence level
        intervals = {}
        
        # The level=0.95 will trigger computation of PDFs which we'll reuse
        first_level = confidence_levels[0]
        lower, upper = self.predict_interval(X, confidence_level=first_level, method=method, num_points=num_points)
        intervals[first_level] = (lower, upper)
        
        # For calibrated models, get the remaining intervals using direct quantile approach
        # This is more consistent with the calibration guarantees
        if self.calibrate and self.is_calibrated:
            for level in confidence_levels[1:]:
                lower, upper = self.predict_interval(X, confidence_level=level, method=method, num_points=num_points)
                intervals[level] = (lower, upper)
        else:
            # For uncalibrated models, we can use the smoothed PDFs for all intervals
            # Get PDFs stored from the first predict_interval call
            pdfs = self._last_pdfs
            
            # For each confidence level (except the first one we already computed)
            for level in confidence_levels[1:]:
                lower_q = (1 - level) / 2
                upper_q = 1 - lower_q
                
                lower_bounds = []
                upper_bounds = []
                
                # Extract intervals from the stored PDFs
                for i, (x_values, pdf_values) in enumerate(pdfs):
                    # Recompute CDF (since it's not stored in _last_pdfs)
                    cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
                    
                    # Normalize CDF to ensure it ends at 1.0
                    if cdf_values[-1] > 0:
                        cdf_values /= cdf_values[-1]
                    
                    # Find the smallest x value such that CDF(x) >= target_probability
                    lower_idx = np.searchsorted(cdf_values, lower_q, side='left')
                    upper_idx = np.searchsorted(cdf_values, upper_q, side='left')
                    
                    # Ensure indices are within bounds
                    lower_idx = max(0, min(lower_idx, len(x_values) - 1))
                    upper_idx = max(0, min(upper_idx, len(x_values) - 1))
                    
                    # Get the x values at those indices
                    lower_bound = x_values[lower_idx]
                    upper_bound = x_values[upper_idx]
                    
                    lower_bounds.append(lower_bound)
                    upper_bounds.append(upper_bound)
                
                intervals[level] = (np.array(lower_bounds), np.array(upper_bounds))
        
        results['intervals'] = intervals
        
        # Use the PDFs that were already computed during predict_interval
        results['pdf'] = self._last_pdfs
        
        return results 
