import json
from pathlib import Path
from collections import defaultdict

def calculate_ground_averages(folder_path, window_size=20):
    ground_scores = defaultdict(list)
    ground_averages = {}

    # Sort files by modification time to process them in chronological order
    files = sorted(Path(folder_path).glob('*.json'), key=lambda x: x.stat().st_mtime)

    for file_path in files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            ground = data['info']['venue']
            
            if 'outcome' in data['info'] and 'method' in data['info']['outcome'] and data['info']['outcome']['method'] == 'D/L':
                print(f"Skipped rain-affected match: {file_path.name}")
                continue
            # Calculate total score correctly
            if data['innings']:
                first_innings = data['innings'][0]
                first_innings_score = 0
                for over in first_innings['overs']:
                    for delivery in over['deliveries']:
                        first_innings_score += delivery['runs']['total']
                
                ground_scores[ground].append(first_innings_score)
            
            # Calculate moving average
            if len(ground_scores[ground]) >= window_size:
                moving_avg = sum(ground_scores[ground][-window_size:]) / window_size
                ground_averages[ground] = round(moving_avg, 2)
            
            print(f"Processed {file_path.name}: Ground - {ground}, Score - {first_innings_score}")
        
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

    # Calculate averages for grounds with fewer than window_size matches
    for ground, scores in ground_scores.items():
        if ground not in ground_averages:
            ground_averages[ground] = round(sum(scores) / len(scores), 2)

    return ground_averages

# Rest of the script remains the same...

# Specify the folder containing your JSON files
folder_path = '/Users/aryamangupta/cric_pred/ipl_it20'

# Calculate ground averages
ground_average_scores = calculate_ground_averages(folder_path)

# Print the results
print("\nGround Average Scores:")
for ground, avg_score in ground_average_scores.items():
    print(f"{ground}: {avg_score}")

# Save the ground averages to a JSON file
with open('ground_average_scores.json', 'w') as f:
    json.dump(ground_average_scores, f, indent=2)

print("\nGround average scores have been saved to 'ground_average_scores.json'")