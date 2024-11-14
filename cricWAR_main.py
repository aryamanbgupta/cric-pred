from cricWAR_data_processor import process_cricket_data
from expected_runs import main_analysis
from regression_models import main_regression_analysis
from war_calculator import main_war_calculation
from war_verify import analyze_2019_results
import glob
import os

def setup_directories():
    """Create necessary output directories"""
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)

def run_cricwar_analysis():
    # Create directories
    setup_directories()
    
    # Paths
    json_files = glob.glob('data/ipl_json/*.json')
    players_info_file = 'data/players_info.csv'
    
    # Step 1: Process raw data
    print("Processing raw data...")
    ball_by_ball_data, venue_stats = process_cricket_data(json_files, players_info_file)
    
    # Step 2: Calculate expected runs and Leverage Index
    print("\nStarting expected runs analysis...")
    final_data, expected_runs, leverage_data = main_analysis(ball_by_ball_data)
    
     # Step 3: Regression Analysis
    print("\nStarting regression analysis...")
    adjusted_data, batting_model, bowling_model = main_regression_analysis(final_data)
    
     # Step 4: Calculate WAR
    # Step 4: Calculate WAR
    print("\nCalculating WAR...")
    player_stats = main_war_calculation(adjusted_data)
    
    # Save results
    print("\nSaving results...")
    player_stats.to_csv('output/data/player_war_stats.csv')
    
    return player_stats, adjusted_data

    '''
    # Save results
    print("\nSaving results...")
    final_data.to_csv('output/data/processed_data.csv', index=False)
    expected_runs.to_csv('output/data/expected_runs.csv', index=False)
    
    return final_data, expected_runs, leverage_data
    '''

if __name__ == "__main__":
    player_stats, adjusted_data = run_cricwar_analysis()
    # Analyze 2019 results specifically
    batting_top, bowling_top, detailed_stats = analyze_2019_results(player_stats, adjusted_data)