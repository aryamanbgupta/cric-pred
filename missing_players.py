import json
import csv
import os

def load_names_csv(file_path):
    names_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            identifier, name = row
            names_dict[identifier] = name
    return names_dict

def extract_players_from_json(json_folder):
    players_dict = {}
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            with open(os.path.join(json_folder, filename), 'r') as jsonfile:
                data = json.load(jsonfile)
                registry = data['info']['registry']['people']
                for name, identifier in registry.items():
                    players_dict[identifier] = name
    return players_dict

def find_missing_players(players_dict, names_dict):
    missing_players = {}
    for identifier, name in players_dict.items():
        if identifier not in names_dict:
            missing_players[identifier] = name
    return missing_players

def save_missing_players_to_csv(missing_players, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['identifier', 'name'])  # Write header
        for identifier, name in missing_players.items():
            writer.writerow([identifier, name])

def create_names_v2_csv(names_dict, missing_players, output_file):
    all_players = {**names_dict, **missing_players}
    sorted_players = sorted(all_players.items(), key=lambda x: x[1].lower())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['identifier', 'name'])  # Write header
        for identifier, name in sorted_players:
            writer.writerow([identifier, name])

def main():
    json_folder = 'ipl_it20'
    names_csv = 'names.csv'
    missing_players_csv = 'missing_players.csv'
    names_v2_csv = 'names_v2.csv'

    names_dict = load_names_csv(names_csv)
    players_dict = extract_players_from_json(json_folder)
    missing_players = find_missing_players(players_dict, names_dict)

    save_missing_players_to_csv(missing_players, missing_players_csv)
    print(f"Missing players have been saved to {missing_players_csv}")

    create_names_v2_csv(names_dict, missing_players, names_v2_csv)
    print(f"All names have been saved in alphabetical order to {names_v2_csv}")

if __name__ == "__main__":
    main()