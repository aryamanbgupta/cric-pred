import csv
import requests
from bs4 import BeautifulSoup
from rpy2 import robjects
from rpy2.robjects.packages import importr
from collections import Counter

# Import the R package that contains find_player_id
r_package = importr('cricketdata')

def get_player_id(player_name):
    result = r_package.find_player_id(player_name)
    player_id = result.rx2('ID')[0]
    return str(int(player_id))

def get_player_styles(player_name):
    player_id = get_player_id(player_name)
    url = f"https://www.espncricinfo.com/cricketers/{player_name.lower().replace(' ', '-')}-{player_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        bowling_style = "Not found"
        batting_style = "Not found"
        
        bowling_style_div = soup.find('p', string='Bowling Style')
        if bowling_style_div:
            bowling_style = bowling_style_div.find_next_sibling('span').text.strip()
        
        batting_style_div = soup.find('p', string='Batting Style')
        if batting_style_div:
            batting_style = batting_style_div.find_next_sibling('span').text.strip()
        
        return bowling_style, batting_style
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}", ""

def process_names_csv(input_file, output_file, tally_file):
    bowling_types = Counter()
    
    with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['identifier', 'name', 'bowling_style', 'batting_style'])
        
        # Skip header in input file
        next(reader)
        
        for row in reader:
            identifier, name = row
            print(f"Processing: {name}")
            bowling_style, batting_style = get_player_styles(name)
            writer.writerow([identifier, name, bowling_style, batting_style])
            
            # Update bowling type tally
            if bowling_style != "Not found" and not bowling_style.startswith("Failed"):
                bowling_types[bowling_style] += 1

    # Write bowling type tally to a separate CSV file
    with open(tally_file, 'w', newline='') as tallyfile:
        tally_writer = csv.writer(tallyfile)
        tally_writer.writerow(['Bowling Style', 'Count'])
        for style, count in bowling_types.items():
            tally_writer.writerow([style, count])

    # Print bowling type tally to console
    print("\nBowling Style Tally:")
    for style, count in bowling_types.items():
        print(f"{style}: {count}")

# Main execution
input_file = 'names_v2.csv'
output_file = 'players_styles.csv'
tally_file = 'bowling_style_tally.csv'

process_names_csv(input_file, output_file, tally_file)
print(f"\nPlayer styles have been saved to {output_file}")
print(f"Bowling style tally has been saved to {tally_file}")