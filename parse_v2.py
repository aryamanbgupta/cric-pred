import json
import numpy as np
from pathlib import Path
import pickle
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import pandas as pd
from collections import defaultdict, Counter
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
                is_wicket = 'wickets' in delivery

                if is_wicket:
                    ball_outcome = -1
                else:
                    ball_outcome = runs
                
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
                    ball_outcome
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
    run_outcomes = []

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
            
            # Collect run outcomes
            for innings in match_sequences:
                run_outcomes.extend([ball[-1] for ball in innings])
            
            print(f"Processed {file_path.name}: {len(match_sequences)} innings, {len(match_player_ids)} players")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    return all_sequences, processed_files, all_player_ids, run_outcomes




# After processing
all_sequences, total_files, unique_player_ids, run_outcomes = process_folder('/Users/aryamangupta/cric_pred/data/ipl_it20')
print(f"\nTotal files processed: {total_files}")
print(f"Total innings sequences collected: {len(all_sequences)}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")

# Count and calculate percentages for run outcomes
total_balls = len(run_outcomes)
outcome_counts = Counter(run_outcomes)
print(f"\nTotal number of balls (sequences for training): {total_balls}")
print("\nDistribution of run outcomes:")
for outcome, count in sorted(outcome_counts.items()):
    percentage = (count / total_balls) * 100
    print(f"  {outcome}: {count} ({percentage:.2f}%)")

# Print data structure
print("\nData structure:")
print(f"Number of innings: {len(all_sequences)}")
print(f"Number of balls in first innings: {len(all_sequences[0])}")



# Find the maximum sequence length
max_len = max(len(seq) for seq in all_sequences)


def pad_3d_sequences(sequences, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    feature_dim = len(sequences[0][0])
    padded_sequences = np.full((len(sequences), max_len, feature_dim), padding_value, dtype=object)
    
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences

# Use the custom function
padded_sequences = pad_3d_sequences(all_sequences)
# After padding
# After padding
#padded_sequences = pad_sequences(all_sequences, maxlen=max_len, padding='post', dtype='object')

# Convert to numpy array if it's not already
padded_sequences = np.array(padded_sequences)

print(f"Shape of padded_sequences: {padded_sequences.shape}")

# Create a mask to identify real data vs padding
# We'll consider a ball as padding if all its features are 0
mask = ~np.all(padded_sequences == 0, axis=2)

# Save the mask
np.save('cricket_sequences_mask.npy', mask)

# Count real balls
real_ball_count = np.sum(mask)
print(f"Number of real balls: {real_ball_count}")

# Count padded balls
padded_ball_count = padded_sequences.shape[0] * padded_sequences.shape[1]
print(f"Number of balls in padded sequences: {padded_ball_count}")

# To get real data (if needed)
real_data = padded_sequences[mask]

print(f"Shape of real_data: {real_data.shape}")

# Save padded sequences
np.save('cricket_sequences.npy', padded_sequences)

# Optionally, save player IDs
with open('unique_player_ids.pkl', 'wb') as f:
    pickle.dump(unique_player_ids, f)

# Save sequences instead of individual balls
#np.save('cricket_sequences.npy', all_sequences)