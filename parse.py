import json
import numpy as np
from pathlib import Path

def get_batting_position(inning, batter):
    batters_order = []
    for over in inning['overs']:
        for delivery in over['deliveries']:
            if delivery['batter'] not in batters_order:
                batters_order.append(delivery['batter'])
    return batters_order.index(batter) + 1

def parse_match_data(json_data):
    data = json.loads(json_data)
    
    # Extract player IDs from the registry
    player_registry = data['info']['registry']['people']
    
    all_balls = []
    
    for inning_idx, inning in enumerate(data['innings'], 1):
        score = 0
        wickets = 0
        balls = 0
        
        for over in inning['overs']:
            for delivery in over['deliveries']:
                batter = delivery['batter']
                bowler = delivery['bowler']
                runs = delivery['runs']['total']
                
                # Use hex IDs instead of integer IDs
                batter_id = player_registry[batter]
                bowler_id = player_registry[bowler]
                
                if 'wicket' in delivery:
                    wickets += 1
                
                ball_input = [
                    inning_idx,
                    score,
                    wickets,
                    balls,
                    batter_id,
                    bowler_id,
                    get_batting_position(inning, batter),
                    runs
                ]
                
                all_balls.append(ball_input)
                
                score += runs
                balls += 1
    
    return all_balls, set(player_registry.values())

def process_folder(folder_path):
    all_balls = []
    all_player_ids = set()
    processed_files = 0

    for file_path in Path(folder_path).glob('*.json'):
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls, match_player_ids = parse_match_data(json_data)
            all_balls.extend(match_balls)
            all_player_ids.update(match_player_ids)
            processed_files += 1
            
            print(f"Processed {file_path.name}: {len(match_balls)} balls, {len(match_player_ids)} players")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    return np.array(all_balls), processed_files, all_player_ids

# Specify the folder containing your JSON files
folder_path = '/Users/aryamangupta/cric_pred/ipl_it20'

training_data, total_files, unique_player_ids = process_folder(folder_path)

print(f"\nTotal files processed: {total_files}")
print(f"Total training examples (balls) collected: {len(training_data)}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")
print("\nFirst few rows of training data:")
print(training_data[:5])

# If you want to save the training data
np.save('cricket_training_data.npy', training_data)

