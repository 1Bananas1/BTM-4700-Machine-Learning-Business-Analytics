import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import os, sys, time

# Selenium imports
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.alert import Alert 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import login info
import logininfo
USER = logininfo.getUser()
PASS = logininfo.getPass()

max_dir = os.path.join(current, "max")
if not os.path.exists(max_dir):
    os.makedirs(max_dir)
    
    
class PricingAgent:
    def __init__(self,data_dir):
        self.best_score = 0
        self.data_dir = data_dir
        self.scores_file = os.path.join(data_dir, 'high_scores.csv')
        self.load_scores()
        
    def load_scores(self):
        """Load high scores history if it exists."""
        if os.path.exists(self.scores_file):
            scores_df = pd.read_csv(self.scores_file)
            if len(scores_df) > 0:
                self.best_score = scores_df['score'].max()
        else:
            pd.DataFrame(columns=['score', 'timestamp', 'screenshot']).to_csv(
                self.scores_file, index=False)
            
    def save_new_score(self, driver, score, screenshot_path):
        """Save a new score and screenshot if it's a high score."""
        if score > self.best_score:
            self.best_score = score
            new_score = pd.DataFrame([{
                'score': score,
                'timestamp': pd.Timestamp.now(),
                'screenshot': screenshot_path
            }])
            filedir = os.path.join(max_dir, f'score_{score:,.2f}.csv')
            output = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="output"]')))
            output.send_keys(Keys.CONTROL,'a')
            output.send_keys(Keys.CONTROL,'c')
            save_cv = pd.read_clipboard()
            save_cv.to_csv(filedir)
            
            if os.path.exists(self.scores_file):
                scores_df = pd.read_csv(self.scores_file)
                scores_df = pd.concat([scores_df, new_score], ignore_index=True)
            else:
                scores_df = new_score
            scores_df.to_csv(self.scores_file, index=False)
            return True
        return False
    
    def predict_price(self, lot_size, rooms, floors, baths):
        risk = 1.092
        base_price = 226841 * risk
        lot_size = float(lot_size)
        rooms = int(rooms)
        floors = int(floors)
        baths = int(baths)
        
        if floors == 1:
            if rooms == 1:
                if lot_size <3000:
                    return 1.28 * base_price
                elif lot_size < 4000:
                    return 1.3 * base_price
                else:
                    return 1.38 * base_price
            
            elif rooms == 2:
                if baths == 1:
                    if lot_size <3000:
                        return 1.22 * base_price
                    elif lot_size < 4000:
                        return 1.24 * base_price
                    else:
                        return 1.16 * base_price
                
                else:
                    if lot_size <3000:
                        return 1.05 * base_price
                    elif lot_size < 4000:
                        return 1.07 * base_price
                    else:
                        return 1.11 * base_price
            
            elif rooms == 3:
                if  baths == 1:
                    if lot_size <3000:
                        return 1.12 * base_price
                    elif lot_size < 4000:
                        return 1.15 * base_price
                    else:
                        return 1.01 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 1.01 * base_price
                    elif lot_size < 4000:
                        return 1.04 * base_price
                    else:
                        return 1.13 * base_price
                
                else:
                    return 1.09 * base_price
            
            elif rooms == 4:
                if  baths == 1:
                    if lot_size <3000:
                        return 1.16 * base_price
                    elif lot_size < 4000:
                        return 1.19 * base_price
                    else:
                        return 1.3 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 0.99 * base_price
                    elif lot_size < 4000:
                        return 1.01 * base_price
                    else:
                        return 1.01 * base_price
                
                else:
                    return 1.09 * base_price
            
            elif rooms == 5:
                if  baths == 1:
                    if lot_size < 3000:
                        return 1.11 * base_price
                    else:
                        return 1.13 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 0.96 * base_price
                    elif lot_size < 4000:
                        return 0.98 * base_price
                    else:
                        return 0.99 * base_price
                
                else:
                    if lot_size <3000:
                        return 0.93 * base_price
                    elif lot_size < 4000:
                        return 0.95 * base_price
                    else:
                        return 0.99 * base_price
            
            elif rooms == 6:
                if  baths == 1:
                    if lot_size < 3000:
                        return 1.11 * base_price
                    else:
                        return 1.13 * base_price
                
                elif baths == 2:
                    if lot_size < 4000:
                        return 0.98 * base_price
                    else:
                        return 1.04 * base_price
                
                else:
                    if lot_size <3000:
                        return 0.89 * base_price
                    elif lot_size < 4000:
                        return 0.9 * base_price
                    else:
                        return 0.96 * base_price
        
        elif floors == 2:
            if rooms == 2:
                if  baths == 1:
                    return base_price
                
                else:
                    return base_price
            
            elif rooms == 3:
                if  baths == 1:
                    if lot_size <3000:
                        return 1.04 * base_price
                    else:
                        return 1.06 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 0.93 * base_price
                    else:
                        return 0.99 * base_price
                
                else:
                    if lot_size <3000:
                        return 0.86 * base_price
                    else:
                        return 0.89 * base_price
            
            elif rooms == 4:
                if  baths == 1:
                    if lot_size <3000:
                        return 1.06 * base_price
                    else:
                        return 1.08 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 0.88 * base_price
                    else:
                        return 0.89 * base_price
                
                else:
                    if lot_size <3000:
                        return 0.86 * base_price
                    else:
                        return 0.88 * base_price
            
            elif rooms == 5:
                if  baths == 1:
                    return base_price
                
                elif baths == 2:
                    return base_price
                
                else:
                    return base_price
            
            elif rooms == 6:
                if  baths == 1:
                    return base_price
                
                elif baths == 2:
                    return base_price
                
                else:
                    return base_price
        
        else:
            if rooms == 3:
                if  baths == 1:
                    if lot_size <3000:
                        return 1.12 * base_price
                    else:
                        return 1.15 * base_price
                
                elif baths == 2:
                    if lot_size <3000:
                        return 0.98 * base_price
                    else:
                        return 1.01 * base_price
                
                else:
                    if lot_size <3000:
                        return 0.92 * base_price
                    else:
                        return 0.94 * base_price
            
            elif rooms == 4:
                if  baths == 1:
                    return base_price * 1.08
                    
                elif baths == 2:
                    return base_price * 1.11
                    
                else:
                    return base_price * 1.15
                          
            elif rooms == 5:
                if  baths == 1:
                    return base_price
                
                elif baths == 2:
                    return base_price
                
                else:
                    return base_price
            
            elif rooms == 6:
                if  baths == 1:
                    return base_price
                
                elif baths == 2:
                    return base_price
                
                else:
                    return base_price
    
    def get_statistics(self):
        """Get current performance statistics."""
        if len(self.history) == 0:
            return "No data available yet."
            
        stats = {
            'Total Attempts': len(self.history),
            'Success Rate': (self.history['saleResult'] == 'Sold').mean() * 100,
            'Total Commission': self.history['commission'].sum(),
            'Average Commission (Successful)': self.history[self.history['saleResult'] == 'Sold']['commission'].mean(),
            'Last 5 Results': self.history.tail()
        }
        return stats
    
def setup_driver():
    """Setup and return the Chrome driver with appropriate options."""
    chromeOptions = Options()
    arguments = [
        "--disable-extensions",
        "--disable-notifications",
        "--disable-infobars",
        "--disable-popup-blocking",
        "--incognito",
        "--blink-settings=imagesEnabled=false"
    ]

    chromeOptions.add_experimental_option("prefs", {
        "download.default_directory": max_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    for arg in arguments:
        chromeOptions.add_argument(arg)

    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chromeOptions)

def login_to_website(driver):
    """Handle website login."""
    try:
        driver.get('https://www.gamelytics.net/real-estate-game.html')
        input_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "wsite-page-membership-text-input"))
        )
        input_element.send_keys(USER + Keys.TAB + PASS + Keys.ENTER)
        output = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="output"]')))
        
        # scroll down
        for j in range(30):
            output.send_keys(Keys.DOWN)
        return True
        
    except Exception as e:
        print(f'Login Error: {e}')
        driver.quit()
        return False

def getStat(driver, **kw):
    """Get statistics from the webpage."""
    keywords = ['lotSize', 'rooms', 'floors', 'baths', 'saleResult','serial','totalCommission']
    if len(kw) != 1:
        raise ValueError('getStat expects exactly one key argument')
    
    keyword = list(kw.keys())[0]
    if keyword not in keywords:
        raise ValueError(f'Invalid getStat keyword. Expected one of: {keywords}')
    
    try:
        stat_value = driver.execute_script(f"""
            var elem = document.querySelector('#{keyword}');
            return elem ? (elem.value || elem.innerText || elem.textContent || '').trim() : '';
        """)
        
        if not stat_value:
            raise ValueError(f"No value found for '{keyword}'.")
        
        return stat_value
    except Exception as e:
        raise ValueError(f"Could not retrieve stat value for {keyword}. Error: {e}")



def run_simulation(driver, max_days=31, tests=1000000):
    """Main simulation loop."""
    agent = PricingAgent(max_dir)
    for test in range(tests):
        for day in range(1, max_days):
            rooms = getStat(driver, rooms='rooms')
            floors = getStat(driver, floors='floors')
            baths = getStat(driver, baths='baths')
            lot_size = getStat(driver, lotSize='lotSize')
            # Get price prediction
            price = agent.predict_price(lot_size, rooms, floors, baths)
            # Set price in UI
            setListingPrice = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="price"]'))
            )
            setListingPrice.clear()  # Clear existing value
            setListingPrice.send_keys(str(round(price,2)))
            
            # Click set price button
            setPrice = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="954388791432399258"]/input[7]'))
            )
            setPrice.click()
            


            
            
            
                
        alert = Alert(driver)
        text = alert.text
        alert.accept()
        score = float(getStat(driver,totalCommission='totalCommission'))
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(max_dir, f'score_{score:,.2f}.png')
        if agent.save_new_score(driver, score, screenshot_path):
            print(f"New high score achieved: ${score:,.2f}!")
            driver.save_screenshot(screenshot_path)
        

            


def main():
    driver = setup_driver()
    
    try:
        if login_to_website(driver):
            run_simulation(driver)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()