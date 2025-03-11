import numpy as np
import pandas as pd


def generate_house_prices_dataset(num_samples=500, random_seed=42):
    """
    Generates a synthetic house prices dataset with numerical and categorical features.

    Parameters:
    - num_samples (int): Number of samples in the dataset.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the generated dataset.
    """
    np.random.seed(random_seed)

    square_feet = np.random.randint(800, 5000, num_samples)
    num_bedrooms = np.random.randint(1, 6, num_samples)
    num_bathrooms = np.random.randint(1, 4, num_samples)
    location_category = np.random.choice(['Urban', 'Suburban', 'Rural'], num_samples)
    year_built = np.random.randint(1950, 2023, num_samples)
    property_condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], num_samples)

    house_price = (
        square_feet * 200
        + num_bedrooms * 10000
        + num_bathrooms * 15000
        + np.where(location_category == 'Urban', 50000, np.where(location_category == 'Suburban', 30000, 10000))
        + np.where(property_condition == 'Excellent', 20000, np.where(property_condition == 'Good', 10000, np.where(property_condition == 'Fair', 0, -10000)))
        + np.random.normal(0, 25000, num_samples)  # Add noise
    )

    df = pd.DataFrame({
        'square_feet': square_feet,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'location_category': location_category,
        'year_built': year_built,
        'property_condition': property_condition,
        'house_price': house_price
    })

    return df

