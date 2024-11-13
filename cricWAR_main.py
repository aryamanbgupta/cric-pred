from cricWAR_data_processor import process_cricket_data
from expected_runs import main_analysis
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
    
    # Save results
    print("\nSaving results...")
    final_data.to_csv('output/data/processed_data.csv', index=False)
    expected_runs.to_csv('output/data/expected_runs.csv', index=False)
    
    return final_data, expected_runs, leverage_data

if __name__ == "__main__":
    final_data, expected_runs, leverage_data = run_cricwar_analysis()