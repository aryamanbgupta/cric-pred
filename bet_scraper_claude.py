from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scroll_to_element(driver, element):
    """Scroll element into view using JavaScript"""
    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(1)  #

try:
    driver = webdriver.Chrome()
    driver.get("https://sportsbook.draftkings.com/sports/cricket")
    wait = WebDriverWait(driver, 10)

    # Get all match format types
    match_types = wait.until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "league-link__link-name"))
    )
    
    # Extract format names
    format_names = [format_type.text for format_type in match_types]
    print(f"Found formats: {format_names}")

    # Navigate through each format
    for format_name in format_names:
        try:
            print(f"\nProcessing {format_name} matches...")
            
            # Click on the format
            format_link = wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, format_name))
            )
            scroll_to_element(driver,format_link)
            format_link.click()
            
            # Wait for the matches to load
            time.sleep(2)  # Consider replacing with explicit wait
            
            # Here you can add code to collect match data, for example:
            bet_categories_elements = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-category-tab-name"))
            )
            bet_category_names = [bet_category.text for bet_category in bet_categories_elements]

            for bet_category_name in bet_category_names:
                print(f"\nProcessing {bet_category_name} category...")
                bet_category_link = wait.until(
                    EC.element_to_be_clickable((By.LINK_TEXT, bet_category_name))
                )
                bet_category_link.click()
                bet_type_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-tabbed-subheader__tab"))
                )
                bet_type_names = [bet_type.text for bet_type in bet_type_elements]

                for bet_type_name in bet_type_names:
                    print(f"\nProcessing {bet_type_name} type...")
                    bet_type_link = wait.until(
                        EC.element_to_be_clickable((By.LINK_TEXT, bet_type_name))
                    )
                    bet_type_link.click()
                    matches_elements = wait.until(
                        EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-event-accordion__title"))
                    )
                    match_names = [match_element.text for match_element in matches_elements]

                    for match in match_names:
                        print(match)



            # try:
            #     # Get all match containers
            #     match_containers = driver.find_elements(By.CLASS_NAME, "sportsbook-outcome-cell__label")
                
            #     # Print team names for each match
            #     print(f"Matches in {format_name}:")
            #     for match in match_containers:
            #         print(f"- {match.text}")
                
            # except Exception as e:
            #     print(f"Error collecting match data: {e}")
            
            # Navigate back to main cricket page
            driver.get("https://sportsbook.draftkings.com/sports/cricket")
            time.sleep(2)  # Allow page to load
            
        except Exception as e:
            print(f"Error processing {format_name}: {e}")
            driver.get("https://sportsbook.draftkings.com/sports/cricket")
            time.sleep(2)
            continue

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()