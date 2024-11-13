import pandas as pd
import numpy as np
from collections import defaultdict
import json

def parse_ball_by_ball_data(json_data, players_info):
    """
    Parse ball-by-ball data into format needed for cricWAR
    
    Parameters:
    json_data: dict - Match data in JSON format
    players_info: DataFrame - Player information including batting/bowling styles
    
    Returns:
    DataFrame - Ball by ball data with required features
    """
    match_info = json_data['info']
    venue = match_info['venue']
    innings_data = []
    
    # Get player registry for IDs
    player_registry = match_info['registry']['people']
    
    for innings_idx, innings in enumerate(json_data['innings'], 1):
        batting_team = innings['team']
        bowling_team = [team for team in match_info['teams'] if team != batting_team][0]
        
        for over in innings['overs']:
            over_num = over['over']
            
            for ball in over['deliveries']:
                # Get player IDs
                batter_id = player_registry[ball['batter']]
                bowler_id = player_registry[ball['bowler']]
                
                # Get runs information
                runs_batter = ball['runs']['batter']
                runs_extras = ball['runs'].get('extras', 0)
                total_runs = ball['runs']['total']
                
                # Extra types
                extras = ball.get('extras', {})
                is_wide = 1 if 'wides' in extras else 0
                is_noball = 1 if 'noballs' in extras else 0
                
                # Wicket information
                is_wicket = 1 if 'wickets' in ball else 0
                
                # Get player information
                batter_info = players_info.get(batter_id, {})
                bowler_info = players_info.get(bowler_id, {})
                
                # Simplify bowling style to pace/spin
                bowling_style = categorize_bowling_style(bowler_info.get('bowling_styles'))
                
                ball_data = {
                    'match_id': f"{match_info['dates'][0]}_{batting_team}_{bowling_team}",
                    'innings': innings_idx,
                    'over': over_num,
                    'ball': len(over['deliveries']),
                    'venue': venue,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'batter': batter_id,
                    'bowler': bowler_id,
                    'batter_handedness': batter_info.get('batting_styles'),
                    'bowling_style': bowling_style,
                    'runs_batter': runs_batter,
                    'runs_extras': runs_extras,
                    'total_runs': total_runs,
                    'is_wide': is_wide,
                    'is_noball': is_noball,
                    'is_wicket': is_wicket
                }
                innings_data.append(ball_data)
                
    return pd.DataFrame(innings_data)

def categorize_bowling_style(style):
    """
    Categorize bowling style as either 'pace' or 'spin'
    
    Parameters:
    style: str - Original bowling style
    
    Returns:
    str - 'pace' or 'spin'
    """
    pace_styles = ['FAST', 'MEDIUM', 'FAST_MEDIUM']  # Add other pace variations
    if any(pace in str(style).upper() for pace in pace_styles):
        return 'pace'
    return 'spin'

def calculate_venue_statistics(matches_data):
    """
    Calculate venue-specific statistics
    
    Parameters:
    matches_data: DataFrame - Ball by ball data for all matches
    
    Returns:
    dict - Venue statistics
    """
    venue_stats = defaultdict(lambda: defaultdict(float))
    
    for venue in matches_data['venue'].unique():
        venue_data = matches_data[matches_data['venue'] == venue]
        
        # Calculate by innings
        for innings in [1, 2]:
            innings_data = venue_data[venue_data['innings'] == innings]
            
            # Calculate phase-wise statistics
            powerplay = innings_data[innings_data['over'] < 6]
            middle = innings_data[(innings_data['over'] >= 6) & (innings_data['over'] < 16)]
            death = innings_data[innings_data['over'] >= 16]
            
            venue_stats[venue].update({
                f'innings_{innings}_avg_score': innings_data.groupby('match_id')['total_runs'].sum().mean(),
                f'innings_{innings}_powerplay_rr': powerplay['total_runs'].mean() * 6,
                f'innings_{innings}_middle_rr': middle['total_runs'].mean() * 6,
                f'innings_{innings}_death_rr': death['total_runs'].mean() * 6
            })
            
        # Calculate chase success rate
        matches = venue_data.groupby('match_id')
        total_matches = len(matches.groups)
        if total_matches > 0:
            successful_chases = 0  # Need to implement chase success logic
            venue_stats[venue]['chase_success_rate'] = successful_chases / total_matches
    
    return dict(venue_stats)

def process_cricket_data(json_files, players_info_file):
    """
    Main processing function for cricWAR data preparation
    
    Parameters:
    json_files: list - List of paths to JSON match files
    players_info_file: str - Path to players info CSV
    
    Returns:
    tuple - (ball_by_ball_data, venue_statistics)
    """
    # Load players info
    players_info = pd.read_csv(players_info_file, index_col='identifier')
    
    # Process all matches
    all_matches_data = []
    for file in json_files:
        with open(file, 'r') as f:
            match_data = json.load(f)
        match_df = parse_ball_by_ball_data(match_data, players_info)
        all_matches_data.append(match_df)
    
    # Combine all matches
    ball_by_ball_data = pd.concat(all_matches_data, ignore_index=True)
    
    # Calculate venue statistics
    venue_stats = calculate_venue_statistics(ball_by_ball_data)
    
    return ball_by_ball_data, venue_stats