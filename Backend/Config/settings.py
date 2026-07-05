from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = BASE_DIR / "Config" / "oanda.cfg"

OANDA_ACCOUNT_ID = os.getenv("ACCOUNT_ID")
OANDA_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
OANDA_ACCOUNT_TYPE = os.getenv("OANDA_ACCOUNT_TYPE")

cfg = BASE_DIR / "Config" / "oanda.cfg"
with open(cfg, "w") as f:

    f.write(f"""
[oanda]

account_id={OANDA_ACCOUNT_ID}

access_token={OANDA_ACCESS_TOKEN}

account_type={OANDA_ACCOUNT_TYPE}
""")