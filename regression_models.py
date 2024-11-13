# regression_models.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# regression_models.py

def prepare_regression_data(data):
    """
    Prepare data for regression models by creating required features
    
    Parameters:
    data: DataFrame - Ball by ball data with leveraged run values
    
    Returns:
    DataFrame - Data with features for regression
    """
    print("\nInitial data shape:", data.shape)
    
    # Create a copy to avoid modifying original data
    regression_data = data.copy()
    
    # Create platoon advantage feature
    def get_platoon_advantage(row):
        if pd.isna(row['batter_handedness']) or pd.isna(row['bowling_style']):
            return 0
        # Right hand batter vs Left arm bowler or vice versa
        return 1 if (('RIGHT' in str(row['batter_handedness']).upper() and 'LEFT' in str(row['bowling_style']).upper()) or
                    ('LEFT' in str(row['batter_handedness']).upper() and 'RIGHT' in str(row['bowling_style']).upper())) else 0
    
    regression_data['platoon_advantage'] = regression_data.apply(get_platoon_advantage, axis=1)
    
    # Check missing values before cleaning
    print("\nMissing values before cleaning:")
    print(regression_data[['venue', 'innings', 'leveraged_run_value', 'platoon_advantage']].isnull().sum())
    
    # Remove rows with NaN in essential columns
    regression_data = regression_data.dropna(subset=['venue', 'innings', 'leveraged_run_value'])
    
    # Check missing values after cleaning
    print("\nMissing values after cleaning:")
    print(regression_data[['venue', 'innings', 'leveraged_run_value', 'platoon_advantage']].isnull().sum())
    
    print("\nFinal regression data shape:", regression_data.shape)
    
    return regression_data

def fit_batting_model(data):
    """
    Fit regression model for batting adjustments
    """
    # Verify data before fitting
    print("\nVerifying batting model data:")
    print("Shape:", data.shape)
    print("Features to be used:", ['venue', 'innings', 'platoon_advantage'])
    print("Sample of data:")
    print(data[['venue', 'innings', 'platoon_advantage', 'leveraged_run_value']].head())
    
    # Prepare features
    categorical_features = ['venue']
    numeric_features = ['innings', 'platoon_advantage']
    
    # Create preprocessing pipeline with updated OneHotEncoder parameters
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Fit model
    X = data[categorical_features + numeric_features]
    y = data['leveraged_run_value']
    
    # Verify no missing values in X and y
    assert X.isnull().sum().sum() == 0, "Missing values found in X"
    assert y.isnull().sum() == 0, "Missing values found in y"
    
    model.fit(X, y)
    
    return model, (categorical_features, numeric_features)


def fit_bowling_model(data):
    """
    Fit regression model for bowling adjustments
    
    Parameters:
    data: DataFrame - Prepared regression data
    
    Returns:
    tuple - (fitted model, feature names)
    """
    # For bowling, we use negative of leveraged run value
    data['bowling_value'] = -data['leveraged_run_value']
    
    # Same features as batting model
    return fit_batting_model(data.rename(columns={'bowling_value': 'leveraged_run_value'}))

def calculate_adjusted_values(data, batting_model, bowling_model, feature_names):
    """
    Calculate adjusted run values using fitted models
    
    Parameters:
    data: DataFrame - Ball by ball data
    batting_model: Pipeline - Fitted batting model
    bowling_model: Pipeline - Fitted bowling model
    feature_names: tuple - Names of features used in models
    
    Returns:
    DataFrame - Data with adjusted run values
    """
    categorical_features, numeric_features = feature_names
    X = data[categorical_features + numeric_features]
    
    # Get predicted values
    batting_predicted = batting_model.predict(X)
    bowling_predicted = bowling_model.predict(X)
    
    # Calculate residuals (adjusted values)
    data['adjusted_batting_value'] = data['leveraged_run_value'] - batting_predicted
    data['adjusted_bowling_value'] = -data['leveraged_run_value'] - bowling_predicted
    
    return data

def main_regression_analysis(data):
    """
    Main function to run regression analysis
    """
    print("Starting regression analysis...")
    print(f"Initial data shape: {data.shape}")
    
    print("\nPreparing regression data...")
    regression_data = prepare_regression_data(data)
    
    print("\nFitting batting model...")
    batting_model, feature_names = fit_batting_model(regression_data)
    
    print("\nFitting bowling model...")
    bowling_model, _ = fit_bowling_model(regression_data)
    
    print("\nCalculating adjusted values...")
    final_data = calculate_adjusted_values(regression_data, batting_model, bowling_model, feature_names)
    
    return final_data, batting_model, bowling_model

def print_model_summary(model, feature_names):
    """
    Print summary statistics for the fitted model
    """
    categorical_features, numeric_features = feature_names
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # Get feature names after preprocessing
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([cat_features, numeric_features])
    
    # Print coefficients
    print("Coefficients:")
    for name, coef in zip(feature_names, regressor.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {regressor.intercept_:.4f}")
    
    # Print R-squared if available
    if hasattr(regressor, 'score'):
        print(f"R-squared: {regressor.score:.4f}")