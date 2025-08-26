'''
File: settings.py
Description:
Created by: Renesh Ravi
'''

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"
LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


DATABASE_PATH = DATA_DIR / "bitcoin_sentiment.db"