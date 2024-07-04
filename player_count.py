import json
import os
from pathlib import Path

def parse_match_data(json_data):
    data = json.loads(json_data)
    ball_count = 0
    player_ids = set()
    
    # Extract player IDs from the registry
    if 'info' in data and 'registry' in data['info'] and 'people' in data['info']['registry']:
        player_ids.update(data['info']['registry']['people'].values())
    
    for inning in data.get('innings', []):
        for over in inning.get('overs', []):
            ball_count += len(over.get('deliveries', []))
    
    return ball_count, player_ids

def process_folder(folder_path):
    total_balls = 0
    processed_files = 0
    all_player_ids = set()

    for file_path in Path(folder_path).glob('*.json'):
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls, match_player_ids = parse_match_data(json_data)
            total_balls += match_balls
            all_player_ids.update(match_player_ids)
            processed_files += 1
            
            print(f"Processed {file_path.name}: {match_balls} balls, {len(match_player_ids)} players")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    return total_balls, processed_files, all_player_ids

# Specify the folder containing your JSON files
folder_path = '/Users/aryamangupta/cric_pred/ipl_it20'

total_training_examples, total_files, unique_player_ids = process_folder(folder_path)

print(f"\nTotal files processed: {total_files}")
print(f"Total training examples (balls) collected: {total_training_examples}")
print(f"Total unique players across all matches: {len(unique_player_ids)}")