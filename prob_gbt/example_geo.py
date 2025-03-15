import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
import alphashape
# Use relative imports for the package
from .prob_gbt import ProbGBT

def create_california_grid(df, training_points, num_points=100):
    """Create a smooth grid of points covering California."""
    # Get the bounds of California from the data with some padding
    lat_min, lat_max = df['latitude'].min() - 0.1, df['latitude'].max() + 0.1
    lon_min, lon_max = df['longitude'].min() - 0.1, df['longitude'].max() + 0.1
    
    # Create a regular grid
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon_min, lon_max, num_points),
        np.linspace(lat_min, lat_max, num_points)
    )
    
    # Convert to DataFrame for prediction
    grid_points = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })
    
    # Create alpha shape from training points (more detailed than convex hull)
    points = [(x, y) for x, y in zip(training_points['longitude'], training_points['latitude'])]
    alpha_shape = alphashape.alphashape(points, alpha=2.0)  # Adjust alpha for detail level
    
    # Filter grid points to only those within the alpha shape
    grid_points['within_bounds'] = grid_points.apply(
        lambda row: alpha_shape.contains(Point(row['longitude'], row['latitude'])),
        axis=1
    )
    grid_points = grid_points[grid_points['within_bounds']].drop('within_bounds', axis=1)
    
    return grid_points, alpha_shape

def plot_california_map(grid_points, values, title, cmap='viridis', 
                       colorbar_label=None, vmin=None, vmax=None,
                       alpha=0.4, figsize=(15, 10), boundary_shape=None):
    """Plot values on a map of California with a base map."""
    # Create a GeoDataFrame from the grid points
    geometry = [Point(xy) for xy in zip(grid_points['longitude'], grid_points['latitude'])]
    gdf = gpd.GeoDataFrame(
        {'values': values},
        geometry=geometry,
        crs="EPSG:4326"  # WGS 84 - standard GPS coordinate system
    )
    
    # Create a bounding box for California with extra padding
    bounds = box(
        grid_points['longitude'].min() - 0.5,
        grid_points['latitude'].min() - 0.5,
        grid_points['longitude'].max() + 0.5,
        grid_points['latitude'].max() + 0.5
    )
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326")
    
    # Convert to Web Mercator projection for the base map
    gdf = gdf.to_crs(epsg=3857)
    bounds_gdf = bounds_gdf.to_crs(epsg=3857)
    
    if boundary_shape is not None:
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_shape], crs="EPSG:4326").to_crs(epsg=3857)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the map extent
    ax.set_xlim(bounds_gdf.geometry.iloc[0].bounds[0], bounds_gdf.geometry.iloc[0].bounds[2])
    ax.set_ylim(bounds_gdf.geometry.iloc[0].bounds[1], bounds_gdf.geometry.iloc[0].bounds[3])
    
    # Add base map first
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom=8
    )
    
    # Plot the boundary shape
    if boundary_shape is not None:
        boundary_gdf.boundary.plot(ax=ax, color='red', linewidth=1, alpha=0.5)
    
    # Plot the values with larger markers
    scatter = ax.scatter(
        gdf.geometry.x, 
        gdf.geometry.y,
        c=values,
        cmap=cmap,
        alpha=alpha,
        s=500,  # larger marker size
        vmin=vmin,
        vmax=vmax,
        edgecolors='white',  # Add white edges to points
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, size=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Customize the plot
    ax.set_title(title, pad=20, size=14)
    ax.set_axis_off()
    
    return fig

def main():
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Load California housing prices dataset
    print("Loading California housing prices dataset...")
    housing_df = pd.read_csv("./prob_gbt/data/california_housing_prices/housing.csv")
    
    # Handle missing values if any
    housing_df = housing_df.dropna()
    
    # For this example, we'll only use latitude and longitude
    X = housing_df[['latitude', 'longitude']]
    y_raw = np.array(housing_df['median_house_value'])
    
    # Apply log transformation to the target variable
    print("Applying log transformation to house prices...")
    y = np.log1p(y_raw)
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize and train the ProbGBT model
    print("\nTraining ProbGBT model...")
    model = ProbGBT(
        num_quantiles=100,
        iterations=3500,
        subsample=1.0,
        random_seed=1234,
        calibrate=True,
        train_separate_models=False
    )

    model.train(
        X_train, 
        y_train,
        eval_set=(X_val, y_val),
        calibration_set=None,
        use_best_model=True,
        verbose=True
    )

    # Create a grid of points covering California
    print("\nCreating prediction grid...")
    grid_points, boundary_shape = create_california_grid(housing_df, X_train, num_points=100)
    
    # Make predictions on the grid
    print("Making predictions on the grid...")
    predictions_log = model.predict(grid_points)
    predictions = np.expm1(predictions_log)
    
    # Get confidence intervals
    print("Computing confidence intervals...")
    lower_bounds_log, upper_bounds_log = model.predict_interval(
        grid_points, confidence_level=0.95, method='spline'
    )
    lower_bounds = np.expm1(lower_bounds_log)
    upper_bounds = np.expm1(upper_bounds_log)
    
    # Calculate relative uncertainty (CI width / prediction)
    relative_uncertainty = (upper_bounds - lower_bounds) / predictions
    
    # Plot predicted house prices
    print("\nCreating visualization plots...")
    fig1 = plot_california_map(
        grid_points, predictions,
        title='Predicted House Prices in California',
        cmap='viridis',
        colorbar_label='Median House Value ($)',
        alpha=0.05,
        boundary_shape=boundary_shape
    )
    fig1.savefig('./images/california_prices_map.png', dpi=300, bbox_inches='tight')
    print("Saved price predictions map to ./images/california_prices_map.png")
    
    # Plot relative uncertainty
    fig2 = plot_california_map(
        grid_points, relative_uncertainty,
        title='Prediction Uncertainty in California\n(95% CI Width / Predicted Price)',
        cmap='RdYlBu_r',  # Red for high uncertainty, blue for low
        colorbar_label='Relative Uncertainty',
        alpha=0.05,
        boundary_shape=boundary_shape
    )
    fig2.savefig('./images/california_uncertainty_map.png', dpi=300, bbox_inches='tight')
    print("Saved uncertainty map to ./images/california_uncertainty_map.png")
    
    # Plot absolute uncertainty
    absolute_uncertainty = upper_bounds - lower_bounds
    fig3 = plot_california_map(
        grid_points, absolute_uncertainty,
        title='Absolute Uncertainty in California\n(95% CI Width in Dollars)',
        cmap='RdYlBu_r',  # Red for high uncertainty, blue for low
        colorbar_label='Confidence Interval Width ($)',
        alpha=0.05,
        boundary_shape=boundary_shape
    )
    fig3.savefig('./images/california_absolute_uncertainty_map.png', dpi=300, bbox_inches='tight')
    print("Saved absolute uncertainty map to ./images/california_absolute_uncertainty_map.png")
    
    # Create a combined visualization
    print("\nCreating combined visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))

    # Create bounding box for California with padding
    bounds = box(
        grid_points['longitude'].min() - 0.5,
        grid_points['latitude'].min() - 0.5,
        grid_points['longitude'].max() + 0.5,
        grid_points['latitude'].max() + 0.5
    )
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:4326").to_crs(epsg=3857)
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_shape], crs="EPSG:4326").to_crs(epsg=3857)
    
    # Set the map extent for all plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(bounds_gdf.geometry.iloc[0].bounds[0], bounds_gdf.geometry.iloc[0].bounds[2])
        ax.set_ylim(bounds_gdf.geometry.iloc[0].bounds[1], bounds_gdf.geometry.iloc[0].bounds[3])
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=8)
        boundary_gdf.boundary.plot(ax=ax, color='red', linewidth=1, alpha=0.5)
    
    # Price predictions
    gdf_prices = gpd.GeoDataFrame(
        {'values': predictions},
        geometry=[Point(xy) for xy in zip(grid_points['longitude'], grid_points['latitude'])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    scatter1 = ax1.scatter(
        gdf_prices.geometry.x,
        gdf_prices.geometry.y,
        c=predictions,
        cmap='viridis',
        alpha=0.05,
        s=500,
        edgecolors='white',
        linewidth=0.5
    )
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Median House Value ($)', size=12)
    ax1.set_title('Predicted House Prices', pad=20, size=14)
    ax1.set_axis_off()
    
    # Relative Uncertainty
    gdf_rel_uncertainty = gpd.GeoDataFrame(
        {'values': relative_uncertainty},
        geometry=[Point(xy) for xy in zip(grid_points['longitude'], grid_points['latitude'])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    scatter2 = ax2.scatter(
        gdf_rel_uncertainty.geometry.x,
        gdf_rel_uncertainty.geometry.y,
        c=relative_uncertainty,
        cmap='RdYlBu_r',
        alpha=0.05,
        s=500,
        edgecolors='white',
        linewidth=0.5
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Relative Uncertainty', size=12)
    ax2.set_title('Relative Uncertainty\n(95% CI Width / Predicted Price)', pad=20, size=14)
    ax2.set_axis_off()
    
    # Absolute Uncertainty
    gdf_abs_uncertainty = gpd.GeoDataFrame(
        {'values': absolute_uncertainty},
        geometry=[Point(xy) for xy in zip(grid_points['longitude'], grid_points['latitude'])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    scatter3 = ax3.scatter(
        gdf_abs_uncertainty.geometry.x,
        gdf_abs_uncertainty.geometry.y,
        c=absolute_uncertainty,
        cmap='RdYlBu_r',
        alpha=0.05,
        s=500,
        edgecolors='white',
        linewidth=0.5
    )
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Confidence Interval Width ($)', size=12)
    ax3.set_title('Absolute Uncertainty\n(95% CI Width in Dollars)', pad=20, size=14)
    ax3.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('./images/california_combined_map.png', dpi=300, bbox_inches='tight')
    print("Saved combined map visualization to ./images/california_combined_map.png")
    
    print("\nExample completed. Check the generated plots in the images directory.")

if __name__ == "__main__":
    # Add the parent directory to the path to make imports work when run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main() 