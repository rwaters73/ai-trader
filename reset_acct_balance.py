import requests
from config import (
    ALPACA_API_KEY_ID as API_KEY,
    ALPACA_API_SECRET_KEY as API_SECRET,
    ALPACA_PAPER_BASE_URL as BASE,
)
#API_KEY = "YOUR_KEY"
#API_SECRET = "YOUR_SECRET"
#BASE = "https://paper-api.alpaca.markets"

# Amount you want as your final balance
TARGET_BALANCE = 2000.00

# Get current balance
account = requests.get(
    f"{BASE}/v2/account",
    headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
).json()

current_cash = float(account["cash"])

# Calculate the difference
adjust_amount = TARGET_BALANCE - current_cash

# Create a journal entry (deposit or withdraw)
data = {
    "entry_type": "JNLC" if adjust_amount < 0 else "JNLD",
    "amount": str(abs(adjust_amount)),
}

resp = requests.post(
    f"{BASE}/v2/account/activities/journal/cash",
    headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET},
    json=data
)

print(resp.status_code, resp.text)
