
# LIBRARY IMPORTS AND CONFIGURATION
import os # For interacting with the operating system, using to check whtether a file exists in my code in this case
import json # For working with JSON data, using to read and write JSON files in my code in this case
import time
from datetime import datetime, timedelta
from typing import List, Dict, Literal # For type hinting, using to specify types of variables and function return types in my code in this case
from io import StringIO # For in-memory text streams, using to handle string data as file-like objects in my code in this case
import config

import numpy as np # For numerical operations, using to handle arrays and mathematical functions 
import pandas as pd # For data manipulation and analysis, using to work with dataframes
import pytz # For timezone handling, using to manage timezones in datetime objects
import requests # For making HTTP requests, using to fetch data from web APIs
from pydantic import BaseModel # For data validation and settings management, using to define data models with type validation

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.enums import DataFeed, Adjustment # For specifying data feed types, using to choose between different data sources in Alpaca API
from alpaca.data.timeframe import TimeFrame # For specifying timeframes, using to define the granularity of historical data in Alpaca API

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce # For specifying order parameters, using to define order side and time in force in Alpaca API


from openai import OpenAI # For interacting with OpenAI's API, using to generate text and analyze sentiment in my code 
import finnhub # For interacting with Finnhub's API, using to fetch financial news and data in my code 

#API KEYS AND CLIENT SETUP
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_API_SECRET = config.ALPACA_API_SECRET
OpenAI_API_KEY = config.OpenAI_API_KEY
ALPACA_PAPER = config.ALPACA_PAPER
FINNHUB_API = config.FINNHUB_API

#STRATEGY PARAMETERS

TZ = pytz.timezone("America/New_York")  # Timezone for market hours

RUN_HOUR = 9 # Hour to run the bot (24-hour format)
RUN_MINUTE = 28 # Minute to run the bot

# ^ Should this just be the time for the Open AI sentiment analysis to be done?
# You know, give it 5 minutes head start to analyze the news before market open?

LOOKBACK_TRADING_DAYS = 126  # Number of trading days to look back for historical data, ~26 weeks. 126 days = ~6 months
SKIP_TRADING_DAYS = 5 # Number of most recent trading days to skip to avoid recent volatility, ~5 days
TARGET_HOLDING = 20  # Target number of stocks I wish to hold in the portfolio, 25
MAX_PER_SECTOR = 5  # Maximum number of stocks to hold per sector to ensure diversification, 6

DATA_FEED = DataFeed.IEX # Tells Alpaca to pull market data from the IEX feed instead of another source

STATE_FILE = "trading_bot_state.json"  # File to store the bot's state

#RISK ASSESSMENT DATA MODELS

class RiskItem(BaseModel): #Creates a new Pydantic model to structure risk assessment data
    symbol: str  # Stock symbol
    risk: Literal["low", "medium", "high"] # Risk level
    trade_block: bool # Whether trading is blocked for this stock
    rationale: str # Rationale for the risk assessment

class RiskReport(BaseModel): #Creates a new Pydantic model to structure the overall risk report
    asof: str # Timestamp of the report, "as of" format
    items: List[RiskItem]  # List of risk items, one risk item per stock

# BOT STATE MANAGEMENT FUNCTIONS

def now_et() -> datetime: #returns datetime object in Eastern Timezone
    return datetime.now(TZ)

def load_state() -> dict: #Loads the bot's state from a JSON file, returns a 'dictionary'
    if not os.path.exists(STATE_FILE): #Checks ti see whether state file exists, if not return empty dictionary
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f: #Opens state file in read mode with UTF-8 encoding
        return json.load(f) #Loads and returns the JSON data as a dictionary
    
def save_state(state: dict) -> None: #Saves the bot's state to a JSON file
    with open(STATE_FILE, "w", encoding="utf-8") as f: #Opens state file in write mode with UTF-8 encoding
        json.dump(state, f, indent=2) #Dumps the state dictionary to the file as JSON with indentation for readability

#DATA FETCHING FUNCTIONS

def get_SP500_companies() -> pd.DataFrame: #Gets list of S&P 500 companies from Wikipedia, returns a pandas DataFrame
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" #URL of the Wikipedia page containing S&P 500 companies
    
    headers = { #Sets custom headers for the HTTP request to mimic a web browser
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    reponse = requests.get(url, headers=headers, timeout=30) #Makes a GET request to the URL with custom headers and a timeout of 30 seconds
    reponse.raise_for_status() #Raises an error if the request was unsuccessful
    tables = pd.read_html(StringIO(reponse.text)) #Parses all HTML tables from the response text using pandas
     #^ Reads all HTML tables from the Wikipedia page into a list of DataFrames
     # ^response.text is the full HTML page downloaded from Wikipedia
     # StringIO allows pandas to read the string as if it were a file
        # pd.read_html returns a list of DataFrames, one per table found in the HTML
    df = tables[0].copy() #Copies the first "table" (aka first DataFrame in list of the webpage), which contains the S&P 500 companies
    df = df.rename(columns={"Symbol": "symbol", "GICS Sector": "sector"}) #Renames columns for easier access
    return df #Returns the DataFrame containing S&P 500 companies


# Fetches Alpaca daily historical close prices for given symbols and returns a DataFrame of 
# closing proces arranged as dates (rows) by symbols (columns)
def fetch_close_matrix(data_client: StockHistoricalDataClient, symbols: List[str]) -> pd.DataFrame:
    end = now_et().astimezone(pytz.UTC) # Gets the current time in ET, then converts it to UTC, as Alpaca API requires UTC timestamps
    start = end - timedelta(days=365) # I set the start date to one year before end to ensure enough data
    print(start) # Checking start date
    print(end) # Checking end date
    req = StockBarsRequest(
        symbol_or_symbols=symbols, # List of stock symbols to fetch data for
        timeframe=TimeFrame.Day, # Daily data
        start=start, # Start date for data fetching
        end=end, # End date for data fetching
        feed=DATA_FEED, # Data feed source, which is set to IEX
        adjustment=Adjustment.ALL, # Price adjustment setting, I set it to ALL so both splits/dividends
    )
    bars = data_client.get_stock_bars(req) # Fetches the stock bars data from Alpaca API
    df = bars.df.copy() # Converts the fetched data to a pandas DataFrame
    #^ Converts the downloaded Alpaca bar data into a pandas Dataframe and makes a safe copy
    # so you can manipulate it without affecting the original data
    if df.empty: #Checks if the DataFrame is empty
        return pd.DataFrame() #Returns an empty DataFrame if no data was fetched
    #This sections turns Alpaca's bar date into a closing price matrix (or "table"), where rows
    # are dates and columns are stock symbols
    # The idea is that close becomes a clean date x symbol DataFrame of closing prices, which I can
    # use for momentum calculations and other analyses later on
    if isinstance(df.index, pd.MultiIndex): #Checks if the DataFrame has a MultiIndex (multiple levels of indexing)
       close = df["close"].reset_index().pivot( #Keeps only the 'close' column, 
           index="timestamp", 
           columns="symbol", 
           values="close"
        ) 
       # ^ Pivots the DataFrame to have timestamps as rows and symbols as columns with closing prices as values
    else:
        close = df.pivot(index="timestamp", columns="symbol", values="close") 
        # ^ Pivots the DataFrame similarly if it doesn't have a MultiIndex
    close = close.sort_index() #Sorts the DataFrame by index (timestamps), putting rows in chronological order (earliest to latest)
    print(close)
    #Earliest goes at the top
    #Latest goes at the bottom
    return close

#MOMENTUM CALCULATION FUNCTIONS

# This function computes each stock's momentum score from the close-price table and returns a ranked list from highest to lowest momentum
def compute_momentum_scores(close: pd.DataFrame) -> pd.Series:
    mom = close.pct_change(periods=LOOKBACK_TRADING_DAYS).shift(SKIP_TRADING_DAYS) 
    print(mom)
    #Computes each stock's percent return over 'LOOKBACK_TRADING_DAYS' 
    #Then lags the signal by  'SKIP_TRADING_DAYS' rows to avoid  short term reversal noise
    latest = mom.iloc[-1].dropna() #Makes a Series (a one-dimensional labeled list of values) with the most recent momentum values for all stocks, dropping 'NaN' values
    latest = latest.replace([np.inf, -np.inf], np.nan).dropna() #Replaces infinite values with NaN and drops them
    print(latest.sort_values(ascending=False))
    return latest.sort_values(ascending=False) #Returns a Pandas Series of all stocks momentum scores in descending value (high -> low)

def pick_top_with_sector_limits(scores: pd.Series, sector_map: Dict[str, str]) -> List[str]:
    picks =[] #List of stock symbols ranked by momentum
    sector_counts: Dict[str, int] = {} #Dictionary to count stocks per sector

    #Loop through each stock symbol in 'scores' (in order), and also grab its score, even though we don't use it here
    for sym, _score in scores.items(): #Iterates over the momentum scores
        sec = sector_map.get(sym, "Unknown") #Given the symbol, its respective sector is given, if no sector is specified, gives "Unknown"
        if sector_counts.get(sec, 0) >= MAX_PER_SECTOR: #Given the sector (Derived from the stock symbol in the line above), conditional checks to see if the value associated with that sector in 'sector_counts'
            # is less than 'MAX_PER_SECTOR' (which is 5)
            continue  # Skip this stock if sector limit is reached
        picks.append(sym) #Adds the stock symbol to the picks list if sector limit isn't exceeded
        sector_counts[sec] = sector_counts.get(sec, 0) + 1 #Increments the count for the sector
        if len(picks) >= TARGET_HOLDING: #Checks if the target number of holdings has been reached
            break  # Stop if we've reached the target number of holdings
    return picks #Returns the list of selected stock symbols


#OPEN AI RISK ASSESSMENT FUNCTION

#Fetches recent news headlines for given stock symbols using Alpaca News API, going back a specified number of days
def fetch_headlines(news_client,  symbols: List[str], days_back: int = 7) -> Dict[str, List[str]]:
    end = pd.Timestamp.now(tz="America/New_York").date() # Defines the most recent date bot should get new articles (whatever day its run)
    start = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=days_back)).date() # Defines the least recent date the bot should get news articles (7 day before the bot was run)
    # Finnhub expects date strings: "YYYY-MM-DD"
    start_str = start.strftime("%Y-%m-%d") 
    end_str = end.strftime("%Y-%m-%d")
    headlines: Dict[str, List[str]] = {s: [] for s in symbols} #Initializes the return value of the Dictionary

    for sym in symbols:
        try: #Runs the following as a safe block as any possible errors thrown don't crash my entire program
            articles = news_client.company_news(sym, _from = start_str, to=end_str) #Call to Finnhub API to get news articles from the specified company based of the given time restraints
        except Exception as e: # If an error is thrown this is relayed instead of crashing my program
            print(f"ERROR fetching news for {sym}: {e}")
            continue
        if not articles: #If 'articles' is empty, the message is relayed to the console 
            print(f"SKIPPED {sym}: no articles returned")
            continue
        for article in articles: #Loops through each item in 'articles' (each 'article' is a dictionary)
            h = (article.get("headline") or "").strip() #Gets the headlines and strips the text to bare necessities
            if not h:
                continue

            if len(headlines[sym]) < 5: # If the number of headlines for the symbol is currently less than 5, than add headlines to the list
                headlines[sym].append(h)
                #print(f"ADDED headline for {sym}: {h}")
            else:
                break
        
    return headlines
        


#Sends each candidate stock's sector and recent headlines to OpenAI
#Returns a dictionary of risk ratings and whether to block trading each stock
def openai_risk_filter(openai_client: OpenAI, candidates: List[str], sector_map: Dict[str, str], headlines: Dict[str, List[str]]) -> Dict[str, RiskItem]:
    #Making one call for all the stocks to be efficient with token usage
    payload = [] #List to hold the payload for OpenAI API
    for s in candidates: #Goes through each stock ticker in the 'candidates' list
        payload.append({
            "symbol": s, # Defines the stock ticker as symbol
            "sector": sector_map.get(s, "Unknown"), #Get the type of sector the stock is in by feeding the ticker into a get call for 'sector_map'
            "headlines": headlines.get(s, []), #Gets the headlines of that stock by feeding the ticker into a get call for 'headlines'
        }) #Appends stock data to the payload
    
    system = ( #System prompt for OpenAI, Python concatenates all the strings into 1 long 'str' sentence
        "You are a risk-check assistant for an automated trading bot. "
        "Your job is NOT to predict returns."
        "Only flag obvious red flags from the provided headlines (fraud allegations, bankruptcy risk, accounting restatement, major lawsuit/regulatory action,"
        " unexpected CEO resignation tied to scandal, severe guidance cuts, etc). "
        "If there are no headlines, or headlines are bland/neutral, do NOT block the trade."
    )

    user = ( #User prompt for OpenAI
        "For each item, output risk (low/medium/high), trade_block(true/false), and a 1-sentence rationale. \n\n"
        f"Data:\n{json.dumps(payload, indent=2)}" #Converts 'payload' list into a JSON string with indents, easier inputs for the model to think
        #User string has instructions and JSON list
    )
    #^ Constructs the user prompt with the payload data formatted as JSON
    resp = openai_client.responses.parse(
         model="gpt-4.1-mini", #Specifies the OpenAI model to use
            input=[
                {"role": "system", "content": system}, #Sets the rules/behavior that I want to models data collection/response to abide by
                {"role": "user", "content": user}, #Gives the model the actual request and data you want it to act on
            ],
            text_format=RiskReport #Tells the OpenAI SDK to make the model's reponse mathc this Pydantic schema and parse the output into a 'RiskReport' object for me
            #SDK stand for "Software Development Kit"
    )
    #^ Sends the request to OpenAI and parses the response into a RiskReport object

    report: RiskReport = resp.output_parsed #Pulls the already parsed, validated results out of the OpenAI Response and stores it in report as a RiskReport Object
    #'report.items' is a list of 'RiskItem' objects
    return {item.symbol: item for item in report.items}  #Builds and returns a dictionary that maps each stock ticker to its 'RiskItem' object


#MARKET TIME UTILITIES

#Waits until the market is open before proceeding, checks every poll_seconds, which defaults to 1 second
def wait_for_market_open(trading_client: TradingClient, poll_seconds: int = 1) -> None:
    while True:
        clock = trading_client.get_clock() #Fetches the current market clock from Alpaca API
        if clock.is_open:
            return
        time.sleep(poll_seconds) #Waits for a specified number of seconds before checking again

def rebalance_equal_weight(trading_client: TradingClient, data_client: StockHistoricalDataClient, target_symbols: List[str]) -> None:
    #test_balance = 0.00
    assets_bought = 0.00
    assets_sold = 0.00
    buffer = 0.999
    min_notional = 1.00
    if not target_symbols: #Checks if the target_symbols list is empty
        print("No target symbols provided for rebalancing, assests stay the same.")
        return
    
    positions = trading_client.get_all_positions() #Fetches all current open positions from Alpaca API
    current = {p.symbol: p for p in positions} #Creates a dictionary mapping symbols to their positions
    sold_any = False #Flag to track if any positions were sold
    print("\n ---ASSETS SOLD ---")
    for sym, pos in current.items(): #Loops through each held position, getting both symbol and position object
        # Position object can be used to get qty, avg_entry_price, etc
        if sym in target_symbols:
            continue  # Keep this position if it's in the target symbols
        qty = float(pos.qty)  #Gets the quantity of shares held for the position, and converts it to a float
        if qty <= 0:
            continue  # Skip if quantity is zero or negative
        latest_trade = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=[sym], feed=DATA_FEED))
        px =float(latest_trade[sym].price) #Gets the latest price of the stock
        est_total = qty * px
        assets_sold += est_total
        print(f"SELL {sym}: qty={qty:.6g}, ref_px=${px:.2f}, est_total=${est_total:.2f}")
        order_request = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(order_request) #Submits the sell order to Alpaca API
        sold_any = True #Sets the sold_any flag to True
    if sold_any:
        print(f"Total assets sold: ${assets_sold:.2f}")
        #test_balance = assets_sold
        print("\nWaiting 10 seconds for sells to settle...")
        time.sleep(10)  # Wait a bit for sells to settle before buying
    
    account = trading_client.get_account() #Fetches the current account information from Alpaca API
    if (sold_any): #If any positions were sold, use the assets_sold as the buying power
        buying_power = float(assets_sold) 
    else:  #Otherwise, use the account's buying power
        buying_power = float(account.cash)
    #test_buying_power = test_balance
    n = len(target_symbols) #Number of target symbols to buy
    w = 1.0/n #Equal weight per stock
    per_stock_notional = round(buying_power * w * buffer, 2) #Calculates the notional amount to allocate per stock
    #test_per_stock_notional = round(test_buying_power * w * buffer, 2)
    print("\n ---ASSETS BOUGHT ---")
    print(f"Buying power=${buying_power:.2f} | targets={n} | per_stock_notional≈${per_stock_notional:.2f}")

    if per_stock_notional < min_notional: #Checks if the per stock notional is less than the minimum notional
        # Basically asking if we have enough buying power to at least buy $1 worth of each stock
        print("Not enough buying power to allocate minimum notional per stock. Aborting buys.")
        return
    
    for sym in target_symbols: #Iterates over each target symbol to buy
        latest_trade = data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=[sym], feed=DATA_FEED)
        )
        px = float(latest_trade[sym].price) #Gets the latest price of the stock
        print(f"BUY  {sym}: notional=${per_stock_notional:.2f}, ref_px=${px:.2f}")
        order_request = MarketOrderRequest(
            symbol=sym,
            notional=per_stock_notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        assets_bought += per_stock_notional
        trading_client.submit_order(order_request) #Submits the buy order to Alpaca API
    print(f"Total assets bought: ${assets_bought:.2f}")

def is_monday(dt: datetime) -> bool:
    return dt.weekday() == 0 #Returns True if the given datetime is a Monday, otherwise False

def week_id(dt: datetime) -> str:
    y, w, _ = dt.isocalendar() #Extracts thez ISO calendar year and week number from the datetime
    return f"{y}-W{w:02d}" #Returns a string in the format "YYYY-Www" representing the week ID

#MAIN TRADING BOT FUNCTION

#This is the main function that runs the trading bot logic once 
def run_once() -> None:
    #Clients initialization
    data_client = StockHistoricalDataClient(
        ALPACA_API_KEY,
        ALPACA_API_SECRET,
    )
    trading_client = TradingClient(
        ALPACA_API_KEY,
        ALPACA_API_SECRET,
        paper=ALPACA_PAPER,
        
    )
    news_client = finnhub.Client(api_key=FINNHUB_API)
    openai_client = OpenAI(api_key=OpenAI_API_KEY)

    #Universe
    uni=get_SP500_companies() #Fetches the list of S&P 500 companies
    symbols = uni["symbol"].tolist() #Extracts the stock symbols from the DataFrame
    sector_map = dict(zip(uni["symbol"], uni["sector"])) #Pairs up the two columns, 'symbol' with 'sector, row by row into tuples like: "('APPL', 'TECH')"
    #'dict' then turns those pairs into a dictionary mapping symbol: 'symbol' -> 'sector'
    # Data + scores
    close = fetch_close_matrix(data_client, symbols) #Fetches the closing price matrix for the stock symbols
    if close.empty or close.shape[0] < (LOOKBACK_TRADING_DAYS + SKIP_TRADING_DAYS + 5): #Checks if there is enough historical data
        #^ If 'close' is empty or if 'close' has less rows (i.e. timestamp dates, each row is a day) than The number of lookback days + the skipping days + 5
        #^ close.shape[0] = rows, close.shape[1] = columns 
        print("Not enough historical data to compute momentum scores.")
        return
    
    scores = compute_momentum_scores(close) #Computes momentum scores for the stocks
    picks = pick_top_with_sector_limits(scores, sector_map) #Selects top stocks with sector limits

    print("\n Top picks before risk assessment:")
    #'i' presents the rank of that stock due to its momentum
    #'s' is the stock symbol, using it to look up its momentum and sector in dictionaries
    for i, s in enumerate(picks, 1): 
        print(f"{i:02d}. {s:6s} mom={scores[s]: .3f} sector={sector_map.get(s)}")

    headlines = fetch_headlines(news_client, picks) #Fetches recent news headlines for the selected stocks
    
    #Prints the headlines of each stock
    print("\n--- HEADLINES BY SYMBOL ---")
    for sym, hl_list in headlines.items():
        print(f"\n{sym}:")
        if not hl_list:
            print("  (no headlines found)")
        else:
            for i, h in enumerate(hl_list, 1):
                print(f"  {i}. {h}")

    risk = openai_risk_filter(openai_client, picks, sector_map, headlines) #Performs risk assessment using OpenAI

    approved = [] #List to hold approved stock symbols after risk assessment
    for s in picks: #Iterates over the selected stock symbols
        item = risk.get(s) #Gets the risk assessment item for the stock symbol
        if item and item.trade_block:
            print(f"BLOCKED {s}: {item.rationale} (risk={item.risk})")
        else:
            approved.append(s) #Adds the stock symbol to the approved list if not blocked

    if len(approved) < 5:
        print("Not enough approved stocks after risk assessment. Aborting trade.")
        return
    
    print("\n Final approved picks after risk assessment:")
    for s in approved:
        item = risk.get(s)
        rat = item.rationale
        print(rat + " ")
        r = item.risk if item else "unknown" #Gets the risk level for the stock symbol
        print(f"- {s} (risk={r})") #Prints the final approved stock symbols with their risk levels
    
    if not is_monday(now_et()):
        print("Today is not Monday, skipping trade execution.")
        return
    
    print("\n Waiting for market to open...")
    wait_for_market_open(trading_client) #Waits until the market is open

    print("Market is open, executing trades...")
    rebalance_equal_weight(trading_client, data_client, approved) #Rebalances the portfolio to equal weight for the approved stocks
    print("Trade execution completed.")


def main_loop() -> None:
    state = load_state() #Loads the bot's state from the state file

    while True:
        t = now_et() #Gets the current time in Eastern Timezone
        if (t.hour == RUN_HOUR and t.minute == RUN_MINUTE) or (t.hour == RUN_HOUR and t.minute == RUN_MINUTE + 1):
            # ^ Allows a 1-minute window to catch the run time
            today = t.strftime("%Y-%m-%d") #Formats the current date as a string
            if state.get("last_daily_run") != today:
                print(f"\n=== Running daily job at {t.isoformat()} ===")
                try:
                    run_once() 
                    state["last_daily_run"] = today #Updates the last daily run date in the state
                    if is_monday(t): #Checks once again to see if it is Monday to rebalance week
                        state["last_rebalance_week"] = week_id(t) #Updates the last rebalance week in the state if today is Monday
                except Exception as e:
                    print("Error during daily run:", str(e)) #Prints any errors that occur during the daily run
                finally:
                    save_state(state) #Saves the updated state to the state file
        else:
            print("Isn't time to run yet or already ran today.")
        time.sleep(10)

#MAIN EXECUTION BLOCK
if __name__ == "__main__":
    main_loop()
