import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from pygam import LinearGAM, s

class ProbGBT:
    """
    Probabilistic Gradient Boosted Trees model that provides uncertainty estimates.
    
    This model uses CatBoost's MultiQuantile loss function to predict multiple quantiles
    of the target distribution, then constructs a probability density function (PDF)
    from these quantiles.
    """
    
    def __init__(self, 
                 num_quantiles=50, 
                 iterations=1000, 
                 learning_rate=None, 
                 depth=None,
                 subsample=1.0,
                 random_seed=42):
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
        """
        self.num_quantiles = num_quantiles
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.subsample = subsample
        self.random_seed = random_seed
        self.model = None
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
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
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
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
        quantile_preds = self.model.predict(X)
        
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
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get quantile predictions
        quantile_preds = self.model.predict(X)
        
        # For a single sample
        if len(quantile_preds.shape) == 1:
            quantile_preds = quantile_preds.reshape(1, -1)
        
        results = []
        
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
        
        return results 