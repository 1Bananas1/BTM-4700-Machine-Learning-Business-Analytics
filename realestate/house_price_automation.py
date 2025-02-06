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

from joblib import dump, load

# Constants
commission_rate = 0.06

# Setup directories

maxcommission_dir = os.path.join(current, "max_commission")
if not os.path.exists(maxcommission_dir):
    os.makedirs(maxcommission_dir)

class HousePricingAgent:
    def __init__(self, data_dir, learning_rate=0.001):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=learning_rate,
            max_iter=2000,
            random_state=42,
            warm_start=True,
            early_stopping=True,  # Add this
            validation_fraction=0.1,  # Add this
            n_iter_no_change=10  # Add this
        )
        self.scaler = StandardScaler()
        self.initialized = False
        self.data_dir = data_dir
        self.history_file = os.path.join(data_dir, 'sales_history.csv')
        self.scores_file = os.path.join(data_dir, 'high_scores.csv')
        self.tests_since_training = 0
        self.best_score = 0
        self.load_history()
        self.load_scores()
        
    def load_history(self):
        """Load previous sales history if it exists."""
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
            if len(self.history) > 0:
                self.train(self.history)
        else:
            self.history = pd.DataFrame(columns=[
                'lotSize', 'rooms', 'floors', 'baths', 
                'price', 'saleResult', 'commission'
            ])
            self.history.to_csv(self.history_file, index=False)
    def save_model(self, filepath):
        """Save the model and scaler state to disk."""
        if not self.initialized:
            print("Model not initialized yet, nothing to save.")
            return
            
        model_path = filepath + '_model.joblib'
        scaler_path = filepath + '_scaler.joblib'
        
        try:
            dump(self.model, model_path)
            dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        """Load the model and scaler state from disk."""
        model_path = filepath + '_model.joblib'
        scaler_path = filepath + '_scaler.joblib'
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load(model_path)
                self.scaler = load(scaler_path)
                self.initialized = True
                print("Model and scaler loaded successfully")
            else:
                print("No saved model found")
        except Exception as e:
            print(f"Error loading model: {e}")

    def load_scores(self):
        """Load high scores history if it exists."""
        if os.path.exists(self.scores_file):
            scores_df = pd.read_csv(self.scores_file)
            if len(scores_df) > 0:
                self.best_score = scores_df['score'].max()
        else:
            pd.DataFrame(columns=['score', 'timestamp', 'screenshot']).to_csv(
                self.scores_file, index=False)

    def save_history(self):
        """Save the updated history to CSV."""
        self.history.to_csv(self.history_file, index=False)

    def save_new_score(self, score, screenshot_path):
        """Save a new score and screenshot if it's a high score."""
        if score > self.best_score:
            self.best_score = score
            new_score = pd.DataFrame([{
                'score': score,
                'timestamp': pd.Timestamp.now(),
                'screenshot': screenshot_path
            }])
            if os.path.exists(self.scores_file):
                scores_df = pd.read_csv(self.scores_file)
                scores_df = pd.concat([scores_df, new_score], ignore_index=True)
            else:
                scores_df = new_score
            scores_df.to_csv(self.scores_file, index=False)
            
            # Save model state when achieving high score
            self.save_model(os.path.join(self.data_dir, f'house_pricing_model_highscore_{score}'))
            return True
        return False

    def preprocess_features(self, data):
        """Preprocess the input features."""
        features = ['lotSize', 'rooms', 'floors', 'baths']
        X = data[features].values
        if not self.initialized:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def train(self, data):
        """Train the model on historical data, prioritizing high commission sales."""
        if len(data) == 0:
            return
            
        # Only consider successful sales
        sold_data = data[data['saleResult'] == 'Sold'].copy()
        if len(sold_data) == 0:
            return
            
        # Calculate the commission percentile for each sale
        sold_data['commission_percentile'] = sold_data['commission'].rank(pct=True)
        
        # Duplicate high-commission sales in the training data
        # Higher commission sales get more duplicates
        training_data = []
        for _, row in sold_data.iterrows():
            # Number of duplicates based on commission percentile (1 to 5 copies)
            n_copies = int(1 + row['commission_percentile'] * 4)
            for _ in range(n_copies):
                training_data.append(row)
        
        training_df = pd.DataFrame(training_data)
        
        X = self.preprocess_features(training_df)
        y = training_df['price'].values
        
        if len(X) > 0:
            self.model.fit(X, y)
            self.initialized = True
            
            # Store the highest successful price for similar properties
            self.max_successful_prices = sold_data.groupby(['rooms', 'floors', 'baths'])['price'].max().to_dict()
    
    def predict_price(self, lot_size, rooms, floors, baths):
        """Predict optimal price for a house, focusing on maximizing commission."""
        features = np.array([[float(lot_size), float(rooms), float(floors), float(baths)]])
        
        if not self.initialized:
            return self._calculate_baseline_price(features)
        
        X = self.scaler.transform(features)
        predicted = self.model.predict(X)[0]
        
        # Get successful sales for similar properties
        similar_sales = self.history[
            (self.history['rooms'] == float(rooms)) &
            (self.history['floors'] == float(floors)) &
            (self.history['baths'] == float(baths)) &
            (self.history['saleResult'] == 'Sold')
        ]
        
        if len(similar_sales) > 0:
            # Find the price that generated the highest commission for similar properties
            max_commission_sale = similar_sales.loc[similar_sales['commission'].idxmax()]
            best_price = max_commission_sale['price']
            
            # Weight between the model prediction and the best historical price
            # As we get more data, trust the model more
            confidence_factor = min(len(similar_sales) / 20, 1.0)
            predicted = (predicted * confidence_factor + best_price * (1 - confidence_factor))
            
            # Add aggressive exploration towards higher prices if we're successful
            if similar_sales['commission'].mean() > similar_sales['commission'].median():
                exploration_factor = np.random.uniform(1.0, 1.15)  # Try up to 15% higher
            else:
                exploration_factor = np.random.uniform(0.95, 1.05)  # Stay within ±5%
        else:
            # If no similar sales, be more conservative but still explore
            exploration_factor = np.random.uniform(0.9, 1.1)
            
        return predicted * exploration_factor
    
    def _calculate_baseline_price(self, features):
        """Calculate baseline price for cold start."""
        lot_size, rooms, floors, baths = features[0]
        if len(self.history) > 0:
            sold_data = self.history[self.history['saleResult'] == 'Sold']
            if len(sold_data) > 0:
                avg_price = sold_data['price'].mean()
                return avg_price
        
        # Fallback to simple heuristic
        return (
            float(lot_size) * 50 +  # Reduced from 100 per sq ft
            float(rooms) * 10000 +  # Reduced from 50000 per room
            float(floors) * 15000 + # Reduced from 75000 per floor
            float(baths) * 5000     # Reduced from 25000 per bath
        )

    def record_result(self, lot_size, rooms, floors, baths, price, sale_result, commission):
        """Record the result of a sale attempt."""
        new_row = pd.DataFrame([{
            'lotSize': float(lot_size),
            'rooms': float(rooms),
            'floors': float(floors),
            'baths': float(baths),
            'price': float(price),
            'saleResult': sale_result,
            'commission': float(commission)
        }])
        
        self.history = pd.concat([self.history, new_row], ignore_index=True)
        self.save_history()

    def maybe_train(self):
        """Train the model every 3 tests and save state."""
        self.tests_since_training += 1
        if self.tests_since_training >= 10 and len(self.history) >= 1:
            self.train(self.history)
            self.tests_since_training = 0
            print("Model retrained after 10 tests")
            # Save model after training
            self.save_model(os.path.join(self.data_dir, 'house_pricing_model'))
            
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
        "download.default_directory": maxcommission_dir,
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

def guessListingPrice(driver, agent, rooms, floors, baths):
    """Predict listing price using the AI agent."""
    try:
        lot_size = getStat(driver, lotSize='lotSize')
        price = agent.predict_price(lot_size, rooms, floors, baths)
        return int(price)  # Round to nearest integer
    except Exception as e:
        print(f"Error in price prediction: {e}")
        return 200000  # Fallback default price

def run_simulation(driver, max_days=31, tests=10000):
    """Main simulation loop."""
    agent = HousePricingAgent(maxcommission_dir)
    model_path = os.path.join(maxcommission_dir, 'house_pricing_model')
    agent.load_model(model_path)  # Will print message if no model exists
    
    for test in range(tests):
        print()
        print(f'Test: {test}/{tests}')
        test_commission = 0  # Track commission for this test
        
        for day in range(1, max_days):
            house = getStat(driver,serial='Serial')
            
            
            
            # Get house features
            rooms = getStat(driver, rooms='rooms')
            floors = getStat(driver, floors='floors')
            baths = getStat(driver, baths='baths')
            lot_size = getStat(driver, lotSize='lotSize')
            time.sleep(0.0125)
            # Get price prediction
            price = guessListingPrice(driver, agent, rooms, floors, baths)
            
            # Set price in UI
            setListingPrice = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="price"]'))
            )
            setListingPrice.clear()  # Clear existing value
            setListingPrice.send_keys(str(price))
            
            # Click set price button
            setPrice = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="954388791432399258"]/input[7]'))
            )
            setPrice.click()
            
            # Small delay to ensure sale result is updated
            time.sleep(0.0125)
            
            
            if day != 30:
                # Get sale result
                saleResult = getStat(driver, saleResult='saleResult')
                # Calculate commission if sold
                commission = price * commission_rate if saleResult == 'Sold' else 0
                test_commission += commission
                agent.record_result(lot_size, rooms, floors, baths, price, saleResult, commission)

        alert = Alert(driver)
        print(alert.text)
        text = alert.text
        alert.accept()
        score = float(getStat(driver,totalCommission='totalCommission'))
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(maxcommission_dir, f'score_{score:,.2f}.png')
        if agent.save_new_score(score, screenshot_path):
            print(f"New high score achieved: ${score:,.2f}!")
            driver.save_screenshot(screenshot_path)
        else:
            print(f"Test completed. Score: ${score:,.2f} (Best: ${agent.best_score:,.2f})")

        agent.maybe_train()
            


def main():
    driver = setup_driver()
    
    try:
        if login_to_website(driver):
            run_simulation(driver)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()