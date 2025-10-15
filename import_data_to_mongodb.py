import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI)
db = client.farm_data # Your database name

crop_yields_collection = db.crop_yields
market_prices_collection = db.market_prices

print("--- Importing Crop Yield Data ---")
# Load crop_yield.csv
try:
    crop_yield_df = pd.read_csv('FertilizerDatset/crop_yield.csv')
    # Ensure 'Year' is an integer
    if 'Year' in crop_yield_df.columns:
        crop_yield_df['Year'] = pd.to_numeric(crop_yield_df['Year'], errors='coerce').fillna(0).astype(int)

    # Convert DataFrame to a list of dictionaries
    crop_yield_data = crop_yield_df.to_dict(orient='records')

    # Clear existing data and insert new data
    crop_yields_collection.delete_many({})
    crop_yields_collection.insert_many(crop_yield_data)
    print(f"Successfully imported {len(crop_yield_data)} documents into 'crop_yields' collection.")
except FileNotFoundError:
    print("Error: FertilizerDatset/crop_yield.csv not found. Skipping crop yield import.")
except Exception as e:
    print(f"Error importing crop yield data: {e}")

print("\n--- Importing Market Prices Data ---")
# Load Price_Agriculture_commodities_Week.csv
try:
    market_prices_df = pd.read_csv('FertilizerDatset/Price_Agriculture_commodities_Week.csv')
    
    # Convert 'date' column to datetime objects if it exists
    if 'date' in market_prices_df.columns:
        market_prices_df['date'] = pd.to_datetime(market_prices_df['date'], errors='coerce')
    
    # Convert DataFrame to a list of dictionaries, handling NaT (Not a Time) for dates
    market_prices_data = market_prices_df.to_dict(orient='records')
    # Filter out any entries where date conversion failed resulting in NaT before inserting
    market_prices_data = [doc for doc in market_prices_data if 'date' not in doc or pd.notna(doc['date'])]

    # Clear existing data and insert new data
    market_prices_collection.delete_many({})
    market_prices_collection.insert_many(market_prices_data)
    print(f"Successfully imported {len(market_prices_data)} documents into 'market_prices' collection.")
except FileNotFoundError:
    print("Error: FertilizerDatset/Price_Agriculture_commodities_Week.csv not found. Skipping market prices import.")
except Exception as e:
    print(f"Error importing market prices data: {e}")

print("\n--- Data Import Complete ---")
