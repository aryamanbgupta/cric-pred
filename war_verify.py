# analysis_2019.py

import pandas as pd
import numpy as np

def load_player_names(filename='names_v2.csv'):
    """
    Load player name mappings from CSV
    """
    names_df = pd.read_csv(filename)
    return dict(zip(names_df['identifier'], names_df['name']))

def analyze_2019_results(player_stats, adjusted_data):
    """
    Analyze and display 2019 results in a format comparable to the paper
    """
    print("\n=== IPL 2019 Analysis ===")
    
    # Load player names mapping
    try:
        player_names = load_player_names()
        print(f"Loaded {len(player_names)} player name mappings")
    except Exception as e:
        print(f"Warning: Could not load player names: {e}")
        player_names = {}
    
    # Get top 8 batting performances
    print("\nTop 8 Batting RAA (Compare with Table 1a in paper):")
    print("=" * 60)
    print(f"{'Player ID':<20} {'Player Name':<25} {'RAA':>10}")
    print("-" * 60)
    batting_top = player_stats.nlargest(8, 'batting_raa')
    for idx, row in batting_top.iterrows():
        name = player_names.get(idx, "Unknown")
        print(f"{idx:<20} {name:<25} {row['batting_raa']:>10.1f}")
    
    # Paper values for reference
    paper_batting = {
        'AD Russell': 120.6,
        'HH Pandya': 82.6,
        'CH Gayle': 72.0,
        'RR Pant': 55.6,
        'PA Patel': 48.3,
        'SP Narine': 36.0,
        'JC Butler': 32.5,
        'JM Bairstow': 28.9
    }
    
    print("\nPaper values for comparison:")
    print("-" * 60)
    for name, raa in paper_batting.items():
        print(f"{'N/A':<20} {name:<25} {raa:>10.1f}")
    
    # Get top 8 bowling performances
    print("\nTop 8 Bowling RAA (Compare with Table 1b in paper):")
    print("=" * 60)
    print(f"{'Player ID':<20} {'Player Name':<25} {'RAA':>10}")
    print("-" * 60)
    bowling_top = player_stats.nlargest(8, 'bowling_raa')
    for idx, row in bowling_top.iterrows():
        name = player_names.get(idx, "Unknown")
        print(f"{idx:<20} {name:<25} {row['bowling_raa']:>10.1f}")
    
    # Paper values for reference
    paper_bowling = {
        'JJ Bumrah': 118.3,
        'JC Archer': 103.3,
        'Rashid Khan': 61.0,
        'B Kumar': 49.1,
        'YS Chahal': 46.8,
        'NA Saini': 44.4,
        'SP Narine': 40.2,
        'R Ashwin': 39.6
    }
    
    print("\nPaper values for comparison:")
    print("-" * 60)
    for name, raa in paper_bowling.items():
        print(f"{'N/A':<20} {name:<25} {raa:>10.1f}")
    
    # Print distribution statistics
    print("\nOverall Statistics:")
    print("=" * 60)
    print("RAA Distribution:")
    print(player_stats[['batting_raa', 'bowling_raa', 'total_raa']].describe())
    
    print("\nWAR Distribution:")
    print(player_stats['war'].describe())
    
    # Save detailed results to CSV
    output_df = player_stats.copy()
    output_df['player_name'] = output_df.index.map(lambda x: player_names.get(x, "Unknown"))
    output_df = output_df.sort_values('war', ascending=False)
    
    # Reorder columns to put name first
    cols = ['player_name'] + [col for col in output_df.columns if col != 'player_name']
    output_df = output_df[cols]
    
    # Save to CSV
    output_df.to_csv('output/data/ipl_2019_detailed_stats.csv')
    print("\nDetailed results saved to 'output/data/ipl_2019_detailed_stats.csv'")
    
    return batting_top, bowling_top, output_df