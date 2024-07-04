import json
import os

def parse_match_data(json_data):
    match_info = json_data['info']
    innings = json_data['innings']
    
    team1 = match_info['teams'][0]
    team2 = match_info['teams'][1]
    #target = match_info['outcome']['by'].get('runs', None)
    
    data_points = []
    
    for inning in innings:
        team_name = inning['team']
        overs = inning['overs']
        score = 0
        
        for over in overs:
            over_number = over['over']
            deliveries = over['deliveries']
            
            for ball_number, delivery in enumerate(deliveries, start=1):
                batter = delivery['batter']
                bowler = delivery['bowler']
                runs = delivery['runs']['total']
                
                score += runs
                
                data_point = {
                    'batter_on_strike': batter,
                    'bowler': bowler,
                    'ball_number': (over_number) * 6 + ball_number,
                    'score_so_far': score,
                    'runs': runs
                    #'target': target
                }
                
                data_points.append(data_point)
                
    return data_points

'''
# Example usage
with open('ipl_json/335982.json', 'r') as f:
    match_data = json.load(f)

data_points = parse_match_data(match_data)

for data_point in data_points:
    print(data_point)
'''


def parse_match_data_from_folder(folder_path):
    data_points = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Load JSON data from the file
            with open(file_path, 'r') as f:
                match_data = json.load(f)
                
                # Parse match data and add to data_points array
                match_data_points = parse_match_data(match_data)
                data_points.extend(match_data_points)

    return data_points

# Specify the folder path containing JSON files
folder_path = 'ipl_json'

# Get data points from all JSON files in the folder
all_data_points = parse_match_data_from_folder(folder_path)

# Print the number of data points
print(f"Total data points: {len(all_data_points)}")

print(all_data_points[1082])

import json

# Export all_data_points to a JSON file
with open('all_data_points.json', 'w') as f:
    json.dump(all_data_points, f)
