def calculate_expected_runs(ball_by_ball_data):
    """
    Calculate expected runs θ(o,w) for each state
    
    Parameters:
    ball_by_ball_data: DataFrame - Ball by ball data with over, wickets, runs info
    
    Returns:
    DataFrame - Expected runs for each state
    """
    # First calculate wickets at each state
    def calculate_wickets_state(group):
        group['wickets'] = group['is_wicket'].cumsum()
        return group
    
    # Apply by match and innings
    ball_by_ball_data = ball_by_ball_data.groupby(['match_id', 'innings']).apply(calculate_wickets_state)
    
    # Calculate empirical average of runs scored in each state
    state_runs = ball_by_ball_data.groupby(['over', 'wickets'])['total_runs'].agg(['mean', 'count']).reset_index()
    state_runs.columns = ['over', 'wickets', 'expected_runs', 'observations']
    
    # Filter states with minimum observations (e.g., 10)
    state_runs = state_runs[state_runs['observations'] >= 10]
    
    return state_runs

def calculate_run_values(ball_by_ball_data, expected_runs):
    """
    Calculate run values (δ) for each ball
    
    Parameters:
    ball_by_ball_data: DataFrame - Ball by ball data
    expected_runs: DataFrame - Expected runs for each state
    
    Returns:
    DataFrame - Ball by ball data with run values
    """
    # Merge expected runs with ball data
    ball_by_ball_data = ball_by_ball_data.merge(
        expected_runs[['over', 'wickets', 'expected_runs']], 
        on=['over', 'wickets'],
        how='left'
    )
    
    # Calculate run value (δ)
    ball_by_ball_data['run_value'] = ball_by_ball_data['total_runs'] - ball_by_ball_data['expected_runs']
    
    return ball_by_ball_data

def calculate_leverage_index(expected_runs):
    """
    Calculate Leverage Index for each state
    
    Parameters:
    expected_runs: DataFrame - Expected runs for each state
    
    Returns:
    DataFrame - Leverage Index for each state
    """
    # Calculate weighted average of expected runs
    weighted_avg = (expected_runs['expected_runs'] * expected_runs['observations']).sum() / expected_runs['observations'].sum()
    
    # Calculate Leverage Index
    expected_runs['leverage_index'] = expected_runs['expected_runs'] / weighted_avg
    
    return expected_runs

def adjust_run_values(ball_by_ball_data, leverage_index):
    """
    Adjust run values by Leverage Index
    
    Parameters:
    ball_by_ball_data: DataFrame - Ball by ball data with run values
    leverage_index: DataFrame - Leverage Index for each state
    
    Returns:
    DataFrame - Ball by ball data with leveraged run values
    """
    # Merge Leverage Index
    ball_by_ball_data = ball_by_ball_data.merge(
        leverage_index[['over', 'wickets', 'leverage_index']], 
        on=['over', 'wickets'],
        how='left'
    )
    
    # Calculate leveraged run value
    ball_by_ball_data['leveraged_run_value'] = ball_by_ball_data['run_value'] / ball_by_ball_data['leverage_index']
    
    return ball_by_ball_data

def main_analysis(ball_by_ball_data):
    """
    Main function to run the expected runs and Leverage Index calculations
    
    Parameters:
    ball_by_ball_data: DataFrame - Processed ball by ball data
    
    Returns:
    DataFrame - Ball by ball data with all calculated metrics
    """
    # Calculate expected runs
    print("Calculating expected runs...")
    expected_runs = calculate_expected_runs(ball_by_ball_data)
    
    # Plot expected runs visualization
    print("\nGenerating expected runs visualization...")
    plot_expected_runs(expected_runs)
    
    # Calculate run values
    print("\nCalculating run values...")
    data_with_run_values = calculate_run_values(ball_by_ball_data, expected_runs)
    
    # Calculate Leverage Index
    print("\nCalculating Leverage Index...")
    leverage_data = calculate_leverage_index(expected_runs)
    
    # Adjust run values by Leverage Index
    print("\nAdjusting run values with Leverage Index...")
    final_data = adjust_run_values(data_with_run_values, leverage_data)
    
    # Validate calculations
    print("\nValidating calculations...")
    validate_calculations(final_data)
    
    return final_data, expected_runs, leverage_data

# Add validation and visualization functions
def validate_calculations(final_data):
    """
    Validate the calculations
    """
    # Check for missing values
    print("Missing values in key columns:")
    print(final_data[['run_value', 'leveraged_run_value', 'leverage_index']].isnull().sum())
    
    # Basic statistics
    print("\nBasic statistics of calculated values:")
    print(final_data[['run_value', 'leveraged_run_value']].describe())
    
    # Verify leverage index properties
    print("\nLeverage Index distribution:")
    print(final_data['leverage_index'].describe())

def plot_expected_runs(expected_runs, save_path='output/figures/'):
    """
    Create visualization of expected runs by state
    
    Parameters:
    expected_runs: DataFrame - Expected runs data
    save_path: str - Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    pivot_data = expected_runs.pivot(index='over', columns='wickets', values='expected_runs')
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Expected Runs by State (Over, Wickets)')
    plt.xlabel('Wickets Lost')
    plt.ylabel('Over')
    
    # Save the plot
    plt.savefig(os.path.join(save_path, 'expected_runs_heatmap.png'))
    plt.close()