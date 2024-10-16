import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session

def check_url(session, url):
    try:
        response = session.get(url, allow_redirects=True)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def find_correct_url(session, base_url):
    years = range(datetime.now().year, 2008, -1)
    for year in years:
        url = f"{base_url}-{year}/results/"
        print(f"Trying URL: {url}")
        response = check_url(session, url)
        if response and response.status_code == 200:
            return url
    return None

def parse_odds(odds_string):
    if odds_string == '-':
        return None
    try:
        return float(odds_string)
    except ValueError:
        print(f"Could not parse odds: {odds_string}")
        return None

def extract_match_data(row):
    columns = row.find_all('td')
    if len(columns) < 7:
        print(f"Row does not have enough columns: {row}")
        return None

    date = columns[0].text.strip()
    teams = columns[1].text.strip().split(' - ')
    if len(teams) != 2:
        print(f"Could not parse teams: {columns[1].text.strip()}")
        return None

    score = columns[2].text.strip()
    home_odds = parse_odds(columns[3].text.strip())
    draw_odds = parse_odds(columns[4].text.strip())
    away_odds = parse_odds(columns[5].text.strip())

    return {
        'Date': date,
        'Home Team': teams[0],
        'Away Team': teams[1],
        'Score': score,
        'Home Odds': home_odds,
        'Draw Odds': draw_odds,
        'Away Odds': away_odds
    }

def scrape_page(session, url):
    response = check_url(session, url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'table-main'})
    if not table:
        print(f"Could not find table with class 'table-main' on {url}")
        print("Page content:")
        print(soup.prettify()[:1000])  # Print first 1000 characters of the page
        return []

    matches = []
    for row in table.find_all('tr', {'class': ['odd', 'even']}):
        match_data = extract_match_data(row)
        if match_data:
            matches.append(match_data)

    if not matches:
        print(f"No matches found on {url}")

    return matches

def save_to_csv(matches, filename):
    if not matches:
        print("No data to save.")
        return

    keys = matches[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(matches)

    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    base_url = "https://www.oddsportal.com/cricket/world/twenty20-international"
    session = create_session()
    correct_url = find_correct_url(session, base_url)
    
    if correct_url:
        print(f"Found correct URL: {correct_url}")
        print(f"Scraping data from {correct_url}...")
        matches = scrape_page(session, correct_url)
        save_to_csv(matches, "twenty20_international_results.csv")
        print(f"Total matches scraped: {len(matches)}")
    else:
        print("Could not find a valid URL for Twenty20 International results.")