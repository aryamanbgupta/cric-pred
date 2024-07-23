import json
import numpy as np
from pathlib import Path
import pickle

with open('ground_average_scores.json', 'r') as f:
    ground_average_scores = json.load(f)

def get_batting_position(inning, batter):
    batters_order = []
    for over in inning['overs']:
        for delivery in over['deliveries']:
            if delivery['batter'] not in batters_order:
                batters_order.append(delivery['batter'])
    return batters_order.index(batter) + 1
    
def parse_match_data(json_data):
    data = json.loads(json_data)
    player_registry = data['info']['registry']['people']
    ground = data['info']['venue']
    default_target = ground_average_scores.get(ground, 160)

    all_sequences = []
    
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
                
                is_extra = 'extras' in delivery
                extra_type = delivery.get('extras', {}).keys()
                is_wide_or_noball = 'wides' in extra_type or 'noballs' in extra_type
                is_noball = 'noballs' in extra_type
                
                ball_input = [
                    inning_idx,
                    score,
                    wickets,
                    balls,
                    batter_id,
                    non_striker_id,
                    bowler_id,
                    get_batting_position(inning, batter),
                    runs,
                    int(is_noball),
                    int(is_wide_or_noball),
                    target
                ]
                
                sequence.append(ball_input)
                
                if 'wicket' in delivery:
                    wickets += 1
                score += runs
                
                if not is_wide_or_noball:
                    balls += 1

        all_sequences.append(sequence)
        if inning_idx == 1:
            first_innings_score = score

    return all_sequences, set(player_registry.values())

# Modify process_folder function to return sequences instead of individual balls
def process_folder(folder_path):
    all_sequences = []
    all_player_ids = set()
    processed_files = 0

    for file_path in Path(folder_path).glob('*.json'):
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

    return all_sequences, processed_files, all_player_ids

# After calling process_folder
all_sequences, total_files, unique_player_ids = process_folder('/Users/aryamangupta/cric_pred/ipl_it20')

print(f"\nTotal files processed: {total_files}")
print(f"Total innings sequences collected: {len(all_sequences)}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")

# Save sequences
np.save('cricket_sequences.npy', all_sequences)

# Optionally, save player IDs
with open('unique_player_ids.pkl', 'wb') as f:
    pickle.dump(unique_player_ids, f)

# Save sequences instead of individual balls
np.save('cricket_sequences.npy', all_sequences)