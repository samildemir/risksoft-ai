import os
import magic
from datetime import datetime, timedelta, timezone
import re
from dotenv import load_dotenv

load_dotenv()

def get_current_time_utc3():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=3)))

def replaceName(text):
        if text is None:
            return None
        turkce_karakterler = "çğıöşüÇĞİÖŞÜ"
        ingilizce_karakterler = "cgiosuCGIOSU"
        ceviri_tablosu = str.maketrans(turkce_karakterler, ingilizce_karakterler)
        text = text.translate(ceviri_tablosu)
        text = text.replace(" ", "_")
        text = re.sub(r'[^A-Za-z0-9_-]', '', text)
        return text

def get_env(key: str, default: str = None) -> str:
    value = os.getenv(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Environment variable {key} not set")
    return value

def detect_file_type( file_content):
        try:
            return magic.Magic(mime=True).from_buffer(file_content)
        except Exception as e:  # Consider specifying the exact exceptions if known
            return None
    
