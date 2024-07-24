import json
import numpy as np
from pathlib import Path
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load player info
players_info = pd.read_csv('data/players_info.csv', index_col='identifier')

# Convert to dictionary for faster lookups
players_info_dict = players_info.to_dict(orient='index')

def get_player_info(player_id):
    info = players_info_dict.get(player_id, {})
    return {
        'batting_style': info.get('batting_styles', 'Unknown'),
        'bowling_style': info.get('bowling_styles', 'Unknown'),
        'playing_role': info.get('playing_roles', 'Unknown')
    }

with open('data/ground_average_scores.json', 'r') as f:
    ground_average_scores = json.load(f)

def get_batting_position(inning, batter):
    batters_order = []
    for over in inning['overs']:
        for delivery in over['deliveries']:
            if delivery['batter'] not in batters_order:
                batters_order.append(delivery['batter'])
    return batters_order.index(batter) + 1

# Dictionary to store player statistics
player_stats = defaultdict(lambda: {'batting': [], 'bowling': []})

def update_player_stats(player_id, role, performance):
    player_stats[player_id][role].append(performance)
    player_stats[player_id][role] = player_stats[player_id][role][-5:]  # Keep only last 5 performances

def get_player_form(player_id, role):
    performances = player_stats[player_id][role]
    if len(performances) < 3:
        return 0  # Not enough data
    return sum(performances[-3:]) / 3  # Average of last 3 performances

def parse_match_data(json_data):
    data = json.loads(json_data)
    player_registry = data['info']['registry']['people']
    ground = data['info']['venue']
    default_target = ground_average_scores.get(ground, 160)

    all_sequences = []

     # Track player performances in this innings
    innings_performances = defaultdict(lambda: {'runs': 0, 'balls_faced': 0, 'wickets': 0, 'runs_conceded': 0})
    
    for inning_idx, inning in enumerate(data['innings'], 1):
        sequence = []
        score = 0
        wickets = 0
        balls = 0
        
        target = default_target if inning_idx == 1 else first_innings_score

        for over in inning['overs']:
            for delivery in over['deliveries']:
                batter = delivery['batter']
                non_striker = delivery['non_striker']
                bowler = delivery['bowler']
                runs = delivery['runs']['total']
                
                batter_id = player_registry[batter]
                non_striker_id = player_registry[non_striker]
                bowler_id = player_registry[bowler]
                
                # Get player info
                batter_info = get_player_info(batter_id)
                non_striker_info = get_player_info(non_striker_id)
                bowler_info = get_player_info(bowler_id)

                # Get form statistics
                batter_form = get_player_form(batter_id, 'batting')
                non_striker_form = get_player_form(non_striker_id, 'batting')
                bowler_form = get_player_form(bowler_id, 'bowling')

                #extras
                is_extra = 'extras' in delivery
                extra_type = delivery.get('extras', {}).keys()
                is_wide_or_noball = 'wides' in extra_type or 'noballs' in extra_type
                is_noball = 'noballs' in extra_type
                is_wicket = 'wicket' in delivery
                
                ball_input = [
                    inning_idx,
                    score,
                    wickets,
                    balls,
                    batter_id,
                    non_striker_id,
                    bowler_id,
                    get_batting_position(inning, batter),
                    batter_info['batting_style'],
                    batter_info['playing_role'],
                    non_striker_info['batting_style'],
                    non_striker_info['playing_role'],
                    bowler_info['bowling_style'],
                    batter_form,
                    non_striker_form,
                    bowler_form,
                    int(is_noball),
                    int(is_wide_or_noball),
                    target,
                    'W' if is_wicket else runs
                ]
                
                sequence.append(ball_input)
                
                # Update innings performances
                innings_performances[batter_id]['runs'] += runs
                innings_performances[batter_id]['balls_faced'] += 1
                innings_performances[bowler_id]['runs_conceded'] += runs
                if is_wicket:
                    innings_performances[bowler_id]['wickets'] += 1
                    wickets += 1
                else:
                    score += runs
                
                if not is_wide_or_noball:
                    balls += 1

        all_sequences.append(sequence)
        if inning_idx == 1:
            first_innings_score = score

         # Update player stats after the innings
        for player_id, performance in innings_performances.items():
            if performance['balls_faced'] > 0:
                batting_performance = performance['runs'] / performance['balls_faced']  # Run rate
                update_player_stats(player_id, 'batting', batting_performance)
            if performance['runs_conceded'] > 0:
                bowling_performance = performance['wickets'] / (performance['runs_conceded'] / 6)  # Wickets per over
                update_player_stats(player_id, 'bowling', bowling_performance)


    return all_sequences, set(player_registry.values())

# Modify process_folder function to return sequences instead of individual balls
def process_folder(folder_path):
    all_sequences = []
    all_player_ids = set()
    processed_files = 0

    # Get all JSON files and sort them by date
    json_files = sorted(
        Path(folder_path).glob('*.json'),
        key=lambda x: json.loads(x.read_text())['info']['dates'][0]
    )

    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_sequences, match_player_ids = parse_match_data(json_data)
            all_sequences.extend(match_sequences)
            all_player_ids.update(match_player_ids)
            processed_files += 1
            
            print(f"Processed {file_path.name}: {len(match_sequences)} innings, {len(match_player_ids)} players")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")
    # Print sample of final processed data
    print("\nSample of processed data:")
    for i, sequence in enumerate(all_sequences[:3]):  # Print first 3 innings
        print(f"\nInnings {i+1}:")
        for j, ball in enumerate(sequence[:5]):  # Print first 5 balls of each innings
            print(f"  Ball {j+1}: {ball}")

    return all_sequences, processed_files, all_player_ids


# After calling process_folder
all_sequences, total_files, unique_player_ids = process_folder('/Users/aryamangupta/cric_pred/data/ipl_it20')

print(f"\nTotal files processed: {total_files}")
print(f"Total innings sequences collected: {len(all_sequences)}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")

# Find the maximum sequence length
max_len = max(len(seq) for seq in all_sequences)

# Pad sequences
padded_sequences = pad_sequences(all_sequences, maxlen=max_len, padding='post', dtype='object')

# Save padded sequences
np.save('cricket_sequences.npy', padded_sequences)

# Optionally, save player IDs
with open('unique_player_ids.pkl', 'wb') as f:
    pickle.dump(unique_player_ids, f)

# Save sequences instead of individual balls
#np.save('cricket_sequences.npy', all_sequences)