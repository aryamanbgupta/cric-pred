from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()

driver.get("https://sportsbook.draftkings.com/sports/cricket")

#driver.find_element(By.CLASS_NAME, "league-link__link-name").click()

#match_types is a list of objects containing all the different match types like test,t20 etc
match_types = driver.find_elements(By.CLASS_NAME, "league-link__link-name")

# print(match_types)
names = [format_type.text for format_type in match_types]

#print(names)

for name in names:
    time.sleep(2)
    driver.find_element(By.LINK_TEXT, name).click()
    time.sleep(3)
    driver.get("https://sportsbook.draftkings.com/sports/cricket")
# for format_type in match_types:
#     print(format_type.text)
#     driver.find_element(By.LINK_TEXT, format_type.text).click()
#     # format_type.click()
#     time.sleep(5)
#     driver.get("https://sportsbook.draftkings.com/sports/cricket")

time.sleep(10)
driver.quit()
