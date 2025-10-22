import requests
import os
import time
import random
import hashlib
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO

PAGE_URL = "https://golestan.ikiu.ac.ir/Forms/AuthenticateUser"
BASE_CAPTCHA = "https://golestan.ikiu.ac.ir/Forms/AuthenticateUser/captcha.aspx"

SAVE_DIR = "Enter a DIR to save images"
os.makedirs(SAVE_DIR, exist_ok=True)
HASHES_FILE = os.path.join(SAVE_DIR, "hashes.txt")

ADDITIONAL = 200
MAX_ATTEMPTS = 5000
DELAY_MIN = 2.5
DELAY_MAX = 5.0
TIMEOUT = 15
RESET_EVERY = 20
BASE_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def is_image_valid(b: bytes) -> bool:
    try:
        img = Image.open(BytesIO(b))
        img.verify()
        return True
    except Exception:
        return False

existing_hashes = set()
if os.path.exists(HASHES_FILE):
    with open(HASHES_FILE, "r", encoding="utf-8") as hf:
        for line in hf:
            h = line.strip()
            if h:
                existing_hashes.add(h)

for p in Path(SAVE_DIR).glob("captcha_*.png"):
    try:
        with open(p, "rb") as f:
            h = md5_bytes(f.read())
        existing_hashes.add(h)
    except Exception:
        pass

start_unique = len(existing_hashes)
target_unique = start_unique + ADDITIONAL
print(f"Loaded {start_unique} existing unique images. Target = {target_unique} total unique images.")

attempt = 0
saved = 0
duplicates = 0
consecutive_duplicates = 0
session = None
session_counter = 0

while len(existing_hashes) < target_unique and attempt < MAX_ATTEMPTS:
    attempt += 1

    if session is None or session_counter >= RESET_EVERY:
        session = requests.Session()
        ua = f"{BASE_UA} r/{random.randint(1000,9999)}"
        session.headers.update({
            "User-Agent": ua,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7"
        })
        session_counter = 0

    try:

        try:
            session.get(PAGE_URL, timeout=TIMEOUT)
        except Exception as e_page:

            print(f"[{attempt}] Warning: couldn't GET PAGE_URL: {e_page} — retry later")
            time.sleep(3)
            continue


        rand_q = random.randint(1_000_000, 9_999_999)
        ts = int(datetime.utcnow().timestamp() * 1000)
        captcha_url = f"{BASE_CAPTCHA}?{ts}_{rand_q}"

        headers = {
            "Referer": PAGE_URL,
            "Cache-Control": "no-cache"
        }

        resp = session.get(captcha_url, headers=headers, timeout=TIMEOUT)
        session_counter += 1

        if resp.status_code != 200:
            print(f"[{attempt}] status={resp.status_code} -> skip")
            time.sleep(2)
            continue

        content = resp.content
        if not content or not is_image_valid(content):
            print(f"[{attempt}] Invalid/empty image -> skip")
            time.sleep(1)
            continue

        h = md5_bytes(content)
        if h in existing_hashes:
            duplicates += 1
            consecutive_duplicates += 1
            print(f"[{attempt}] Duplicate (md5={h[:8]}). duplicates={duplicates}. consec_dup={consecutive_duplicates}")

            if consecutive_duplicates >= 3:
                backoff = random.uniform(8, 12)
                print(f"   consecutive duplicates >=3 -> sleeping extra {backoff:.1f}s")
                time.sleep(backoff)
        else:

            filename = os.path.join(SAVE_DIR, f"captcha_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]}_{len(existing_hashes)+1}.png")
            with open(filename, "wb") as f:
                f.write(content)

            with open(HASHES_FILE, "a", encoding="utf-8") as hf:
                hf.write(h + "\n")
            existing_hashes.add(h)

            saved += 1
            consecutive_duplicates = 0
            print(f"[{attempt}] Saved UNIQUE #{len(existing_hashes)}: {os.path.basename(filename)} (md5={h[:8]})")

        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)

    except Exception as e:
        print(f"[{attempt}] Exception: {e}")
        time.sleep(3)

print("---- Done ----")
print(f"Attempts: {attempt}, Saved this run: {saved}, Total unique now: {len(existing_hashes)}, Duplicates seen: {duplicates}")
print(f"Saved images directory: {os.path.abspath(SAVE_DIR)}")
print(f"Hashes file: {os.path.abspath(HASHES_FILE)}")
if len(existing_hashes) < target_unique:
    print(f"Note: stopped before reaching target ({target_unique}) — consider increasing MAX_ATTEMPTS or DELAY_MAX, or check server-side limits.")
else:
    print(f"Success: collected {ADDITIONAL} new unique images.")
