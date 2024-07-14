import requests
from bs4 import BeautifulSoup

def get_bowling_style(player_name, player_id):
    # Construct the URL
    url = f"https://www.espncricinfo.com/cricketers/{player_name}-{player_id}"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the div containing "Bowling Style"
        bowling_style_div = soup.find('p', string='Bowling Style')
        
        if bowling_style_div:
            # Get the next sibling which contains the actual bowling style
            bowling_style = bowling_style_div.find_next_sibling('span').text.strip()
            return bowling_style
        else:
            return "Bowling style not found"
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"

# Example usage
player_name = "m-pathirana"
player_id = "1194795"

bowling_style = get_bowling_style(player_name, player_id)
print(f"Bowling Style: {bowling_style}")