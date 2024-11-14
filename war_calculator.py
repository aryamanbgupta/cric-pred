# war_calculator.py
import pandas as pd

def identify_replacement_level(data):
    """
    Identify replacement level players based on playing time
    Following the paper's approach: 8 batters and 8 bowlers per team as league-level players
    """
    print("\nStarting replacement level identification...")
    
    # Get basic player participation stats
    player_stats = pd.DataFrame({
        'balls_batted': data.groupby('batter').size(),
        'balls_bowled': data.groupby('bowler').size()
    }).fillna(0)
    
    # Get number of teams
    num_teams = len(data['batting_team'].unique())
    print(f"Number of teams identified: {num_teams}")
    
    # Calculate threshold for league-level players (8 per team per role)
    num_league_players = 8 * num_teams
    print(f"Target number of league-level players per role: {num_league_players}")
    
    # Identify league-level players
    league_batters = set(player_stats.nlargest(num_league_players, 'balls_batted').index)
    league_bowlers = set(player_stats.nlargest(num_league_players, 'balls_bowled').index)
    league_players = league_batters.union(league_bowlers)
    
    # Identify replacement players
    all_players = set(player_stats.index)
    replacement_players = all_players - league_players
    
    print(f"\nPlayer Classification:")
    print(f"Total Players: {len(all_players)}")
    print(f"League Batters: {len(league_batters)}")
    print(f"League Bowlers: {len(league_bowlers)}")
    print(f"Total League Players: {len(league_players)}")
    print(f"Replacement Players: {len(replacement_players)}")
    
    return league_players, replacement_players, player_stats

def calculate_replacement_level_RAA(data, replacement_players):
    """
    Calculate average RAA per ball for replacement level players
    """
    print("\nCalculating replacement level RAA...")
    
    # Get replacement player data
    replacement_data = data[
        (data['batter'].isin(replacement_players)) | 
        (data['bowler'].isin(replacement_players))
    ]
    
    # Calculate averages
    replacement_batting = replacement_data['adjusted_batting_value'].mean()
    replacement_bowling = replacement_data['adjusted_bowling_value'].mean()
    
    print(f"\nReplacement Level Metrics:")
    print(f"Batting RAA per ball: {replacement_batting:.4f}")
    print(f"Bowling RAA per ball: {replacement_bowling:.4f}")
    
    # Use average of batting and bowling
    avg_replacement_RAA = (replacement_batting + replacement_bowling) / 2
    print(f"Average RAA per ball: {avg_replacement_RAA:.4f}")
    
    return avg_replacement_RAA

def calculate_player_RAA(data, player_stats):
    """
    Calculate RAA for each player
    """
    print("\nCalculating player RAA values...")
    
    # Calculate batting RAA
    batting_raa = data.groupby('batter').agg({
        'adjusted_batting_value': ['sum', 'count']
    })
    batting_raa.columns = ['batting_raa', 'batting_balls']
    
    # Calculate bowling RAA
    bowling_raa = data.groupby('bowler').agg({
        'adjusted_bowling_value': ['sum', 'count']
    })
    bowling_raa.columns = ['bowling_raa', 'bowling_balls']
    
    # Combine statistics
    player_raa = pd.DataFrame(index=player_stats.index)
    player_raa['batting_raa'] = batting_raa['batting_raa']
    player_raa['batting_balls'] = batting_raa['batting_balls']
    player_raa['bowling_raa'] = bowling_raa['bowling_raa']
    player_raa['bowling_balls'] = bowling_raa['bowling_balls']
    
    # Fill NaN values with 0
    player_raa = player_raa.fillna(0)
    
    # Calculate totals
    player_raa['total_raa'] = player_raa['batting_raa'] + player_raa['bowling_raa']
    player_raa['total_balls'] = player_raa['batting_balls'] + player_raa['bowling_balls']
    
    print("\nRAA Summary Statistics:")
    print(player_raa[['total_raa', 'batting_raa', 'bowling_raa']].describe())
    
    return player_raa

def calculate_WAR(player_raa, avg_replacement_RAA):
    """
    Calculate WAR for each player
    """
    print("\nCalculating WAR...")
    
    player_stats = player_raa.copy()
    
    # Calculate VORP (Value Over Replacement Player)
    player_stats['replacement_level_raa'] = player_stats['total_balls'] * avg_replacement_RAA
    player_stats['vorp'] = player_stats['total_raa'] - player_stats['replacement_level_raa']
    
    # Calculate WAR (using RPW from paper - 84.5)
    RPW = 84.5  # Runs Per Win value from the paper
    player_stats['war'] = player_stats['vorp'] / RPW
    
    print("\nWAR Summary Statistics:")
    print(player_stats['war'].describe())
    
    # Print top players
    print("\nTop 10 Players by WAR:")
    print(player_stats.nlargest(10, 'war')[['war', 'total_raa', 'total_balls']])
    
    return player_stats

def main_war_calculation(adjusted_data):
    """
    Main function to run WAR calculations
    """
    print("Starting WAR calculations...")
    
    # Step 1: Identify replacement level
    league_players, replacement_players, player_stats = identify_replacement_level(adjusted_data)
    
    # Step 2: Calculate replacement level RAA
    avg_replacement_RAA = calculate_replacement_level_RAA(adjusted_data, replacement_players)
    
    # Step 3: Calculate player RAA
    player_raa = calculate_player_RAA(adjusted_data, player_stats)
    
    # Step 4: Calculate WAR
    final_stats = calculate_WAR(player_raa, avg_replacement_RAA)
    
    return final_stats