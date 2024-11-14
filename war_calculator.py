# war_calculator.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def identify_replacement_level(data, season):
    """
    Identify replacement level players based on playing time
    Following the paper's approach: 8 batters and 8 bowlers per team as league-level players
    
    Parameters:
    data: DataFrame - Ball by ball data with player statistics
    season: str - Season identifier
    
    Returns:
    tuple - (set of league-level player IDs, set of replacement-level player IDs)
    """
    # Count number of balls played/bowled by each player
    player_stats = pd.DataFrame({
        'balls_batted': data.groupby('batter').size(),
        'balls_bowled': data.groupby('bowler').size()
    }).fillna(0)
    
    # Get number of teams in the season
    num_teams = len(data['batting_team'].unique())
    num_league_players = 8 * num_teams  # 8 players per team for each role
    
    # Identify league-level players
    league_batters = set(player_stats.nlargest(num_league_players, 'balls_batted').index)
    league_bowlers = set(player_stats.nlargest(num_league_players, 'balls_bowled').index)
    league_players = league_batters.union(league_bowlers)
    
    # All other players are replacement level
    all_players = set(player_stats.index)
    replacement_players = all_players - league_players
    
    return league_players, replacement_players

def calculate_replacement_level_RAA(data, replacement_players):
    """
    Calculate average RAA per ball for replacement level players
    
    Parameters:
    data: DataFrame - Ball by ball data with adjusted values
    replacement_players: set - Set of replacement level player IDs
    
    Returns:
    float - Average RAA per ball for replacement level players
    """
    replacement_data = data[
        (data['batter'].isin(replacement_players)) | 
        (data['bowler'].isin(replacement_players))
    ]
    
    # Calculate average RAA per ball for replacement players
    batting_raa = replacement_data['adjusted_batting_value'].mean()
    bowling_raa = replacement_data['adjusted_bowling_value'].mean()
    
    return (batting_raa + bowling_raa) / 2

def calculate_player_RAA(data, avg_replacement_RAA):
    """
    Calculate RAA for each player
    
    Parameters:
    data: DataFrame - Ball by ball data with adjusted values
    avg_replacement_RAA: float - Average RAA per ball for replacement level
    
    Returns:
    DataFrame - Player-level RAA statistics
    """
    # Calculate batting RAA
    batting_raa = data.groupby('batter').agg({
        'adjusted_batting_value': ['sum', 'count'],
        'batter': 'first'  # To keep player ID
    })
    
    # Calculate bowling RAA
    bowling_raa = data.groupby('bowler').agg({
        'adjusted_bowling_value': ['sum', 'count'],
        'bowler': 'first'  # To keep player ID
    })
    
    # Combine batting and bowling RAA
    player_raa = pd.DataFrame({
        'batting_raa': batting_raa[('adjusted_batting_value', 'sum')],
        'batting_balls': batting_raa[('adjusted_batting_value', 'count')],
        'bowling_raa': bowling_raa[('adjusted_bowling_value', 'sum')],
        'bowling_balls': bowling_raa[('adjusted_bowling_value', 'count')]
    }).fillna(0)
    
    # Calculate total RAA
    player_raa['total_raa'] = player_raa['batting_raa'] + player_raa['bowling_raa']
    player_raa['total_balls'] = player_raa['batting_balls'] + player_raa['bowling_balls']
    
    # Calculate VORP
    player_raa['replacement_level_raa'] = player_raa['total_balls'] * avg_replacement_RAA
    player_raa['vorp'] = player_raa['total_raa'] - player_raa['replacement_level_raa']
    
    return player_raa

def calculate_runs_per_win(data, seasons):
    """
    Calculate runs per win using regression approach from the paper
    
    Parameters:
    data: DataFrame - Ball by ball data
    seasons: list - List of seasons to include
    
    Returns:
    float - Runs per win value
    """
    # Calculate run differential and wins for each team in each season
    team_stats = []
    
    for season in seasons:
        season_data = data[data['season'] == season]
        
        for team in season_data['batting_team'].unique():
            # Calculate run differential
            runs_scored = season_data[season_data['batting_team'] == team]['total_runs'].sum()
            runs_allowed = season_data[season_data['bowling_team'] == team]['total_runs'].sum()
            run_diff = runs_scored - runs_allowed
            
            # Calculate wins
            team_matches = season_data[
                (season_data['batting_team'] == team) | 
                (season_data['bowling_team'] == team)
            ]['match_id'].unique()
            
            wins = sum(1 for match in team_matches if 
                      data[(data['match_id'] == match) & 
                           (data['innings'] == 2)]['batting_team'].iloc[-1] == team)
            
            team_stats.append({
                'season': season,
                'team': team,
                'run_diff': run_diff,
                'wins': wins
            })
    
    team_stats_df = pd.DataFrame(team_stats)
    
    # Fit regression model
    model = LinearRegression()
    model.fit(team_stats_df[['run_diff']], team_stats_df['wins'])
    
    # RPW is 1/coefficient
    rpw = 1 / model.coef_[0]
    
    return rpw

def calculate_WAR(player_raa, rpw):
    """
    Calculate WAR for each player
    
    Parameters:
    player_raa: DataFrame - Player-level RAA statistics
    rpw: float - Runs per win value
    
    Returns:
    DataFrame - Player statistics including WAR
    """
    player_stats = player_raa.copy()
    player_stats['war'] = player_stats['vorp'] / rpw
    
    return player_stats

def main_war_calculation(adjusted_data, seasons):
    """
    Main function to calculate WAR
    
    Parameters:
    adjusted_data: DataFrame - Ball by ball data with adjusted values
    seasons: list - List of seasons to include
    
    Returns:
    DataFrame - Player statistics including WAR
    """
    print("Starting WAR calculations...")
    
    # Identify replacement level players
    league_players, replacement_players = identify_replacement_level(adjusted_data, seasons)
    print(f"\nIdentified {len(league_players)} league-level players and {len(replacement_players)} replacement-level players")
    
    # Calculate replacement level RAA
    avg_replacement_RAA = calculate_replacement_level_RAA(adjusted_data, replacement_players)
    print(f"\nAverage replacement level RAA per ball: {avg_replacement_RAA:.4f}")
    
    # Calculate player RAA
    player_raa = calculate_player_RAA(adjusted_data, avg_replacement_RAA)
    print("\nCalculated RAA for all players")
    
    # Calculate runs per win
    rpw = calculate_runs_per_win(adjusted_data, seasons)
    print(f"\nCalculated runs per win: {rpw:.2f}")
    
    # Calculate WAR
    player_stats = calculate_WAR(player_raa, rpw)
    print("\nCalculated WAR for all players")
    
    # Print summary statistics
    print("\nWAR Summary Statistics:")
    print(player_stats['war'].describe())
    
    return player_stats