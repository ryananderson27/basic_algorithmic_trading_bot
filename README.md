Automated Weekly Trading Bot
By: Ryan Anderson

DISCLAIMER: This repo and subsequent code are for educational purposes/paper trading use only, not finanical advice. Don't use this bot to trade stocks with any actual currency

BOT OPERATIONS
- Pulls the current list of all companies in the S&P 500 (Takes from Wikipedia)
- Using the Alpaca API to gather historical data, it computes every single companies 6 month momentum
- Selects the top 20 stocks with the highest momentum, with a sector cap of 5 max for diversification
- Pulls recent news headlines, from Alpaca API, for the selected stocks
- Uses the OpenAI API to flag obvious “do-not-trade” red flags from the headlines.
- If enough stocks pass the risk filter, it waits for the market to open and then:
    - Sells positions that are not in the approved list
    - Buys the approved list at equal weight using notional orders
- Repeat every week on Monday, 9:27 AM EST

SETUP (WINDOWS):

1. Folder/File Layout

Create a project folder like this:

trading-bot/trading_bot.py
trading-bot/config.py

trading_bot.py is where your main bot code will go
config.py is were your API keys will go

I use VS Code and its terminal to make this code, I would recommend you do the same

2. Python Installation

Code uses Python 3.14.2, so install, at the least, that version, and check using 'py --version'

3. Creating a Virtual Environment

Witin the 'trading-bot' folder you want to make a Virtual Environment
Using 'powershell' navigate yourself to that folder, and then input the following:

py -m venv .venv
.venv\Scripts\activate

This will activate your virtual environment

4. Install Dependencies

With venv activated, input the following prompts into your 'powershell':

pip install alpaca-py openai pandas numpy requests pydantic pytz

This will install the necessary libraries you need for the bot to run

5. API SETUP

Alpaca API (Paper Trading):
- Create/log into your Alpaca account
- Get your Paper Trading API Key + Secret
- Put them into config.py

OpenAI API 
- Create an OpenAI API key from your OpenAI account
- Put it into config.py 

DISCLAIMER: For me I had to pay OpenAI to use API tokens, paid like $10

Your 'config.py' file should look like this:

# config.py

ALPACA_API_KEY = "YOUR_ALPACA_KEY"
ALPACA_API_SECRET = "YOUR_ALPACA_SECRET"

OpenAI_API_KEY = "YOUR_OPENAI_KEY"

# Keep paper trading on while testing
ALPACA_PAPER = True

DON'T SHARE THIS FILE WITH YOUR API KEYS

6. BOT CODE

Either copy/paste or make a Git pull request from the repo into your trading_bot python file.
Now you have the code. Make sure everything is properly imported and there are no missing libraries.
Make sure your API keys import properly also.

7. RUNNING THE BOT

Just run:

'python trading_bot.py"
