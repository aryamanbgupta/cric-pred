import json
import os
from pathlib import Path

def parse_match_data(json_data):
    data = json.loads(json_data)
    ball_count = 0
    
    for inning in data.get('innings', []):
        for over in inning.get('overs', []):
            ball_count += len(over.get('deliveries', []))
    
    return ball_count

def process_folder(folder_path):
    total_balls = 0
    processed_files = 0

    for file_path in Path(folder_path).glob('*.json'):
        try:
            with open(file_path, 'r') as file:
                json_data = file.read()
            
            match_balls = parse_match_data(json_data)
            total_balls += match_balls
            processed_files += 1
            
            print(f"Processed {file_path.name}: {match_balls} balls")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    return total_balls, processed_files

# Specify the folder containing your JSON files
#folder_path = '/Users/aryamangupta/cric_pred/t20s_male_json'
folder_path = '/Users/aryamangupta/cric_pred/ipl_json'


total_training_examples, total_files = process_folder(folder_path)

print(f"\nTotal files processed: {total_files}")
print(f"Total training examples (balls) collected: {total_training_examples}")