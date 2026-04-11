# =============================
# IMPORT LIBRARIES
# =============================

# Flask → create API server
from flask import Flask, request, jsonify

# PIL → image processing
from PIL import Image, ImageOps

# EasyOCR → extract text from images
import easyocr

# io → handle image bytes
import io

# os → file paths
import os

# torch → used by EasyOCR (AI backend)
import torch

# numpy → image arrays
import numpy as np

# regex → text processing
import re

# pandas → read CSV
import pandas as pd

# fuzzy matching → compare text similarity
from rapidfuzz import fuzz

# barcode reader
from pyzbar.pyzbar import decode as zbar_decode


# =============================
# CREATE FLASK APP
# =============================
app = Flask(__name__)

# Limit CPU usage (important for Render)
torch.set_num_threads(1)


# =============================
# PATH CONFIGURATION
# =============================

# Get current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where EasyOCR models are stored
MODEL_DIR = os.environ.get("EASYOCR_MODULE_PATH", os.path.join(BASE_DIR, ".EasyOCR"))

# CSV file path
CSV_FILE_PATH = os.path.join(BASE_DIR, "items.csv")


# =============================
# LOAD OCR MODEL (AI)
# =============================
reader = easyocr.Reader(
    ['en'],                # language = English
    gpu=False,             # CPU only
    model_storage_directory=MODEL_DIR,  # where model stored
    download_enabled=True  # download model if not exists
)


# =============================
# IMAGE PROCESSING
# =============================

# Convert bytes → image
def open_image_from_bytes(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


# Improve image quality for OCR
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img = ImageOps.autocontrast(img)  # improve contrast
    return np.array(img)


# Extract text from image using OCR
def extract_text_from_bytes(file_bytes):
    img = preprocess_image(file_bytes)

    # OCR reading
    results = reader.readtext(img, detail=0, paragraph=True)

    # combine all detected text
    full_text = " ".join(results).strip()

    return full_text, results


# =============================
# BARCODE DETECTION
# =============================

# Clean barcode text
def clean_barcode_text(s):
    if not s:
        return ""
    s = str(s).strip()
    s = s.replace("\n", "")
    s = s.replace("\r", "")
    s = re.sub(r"\s+", "", s)
    return s


# Read barcode from image
def decode_barcodes_from_bytes(file_bytes):
    img = open_image_from_bytes(file_bytes)

    # decode barcode
    decoded = zbar_decode(img)

    barcodes = []

    for obj in decoded:
        try:
            data = obj.data.decode("utf-8", errors="ignore").strip()
        except:
            data = str(obj.data)

        barcodes.append({
            "type": str(obj.type),   # barcode type
            "data": clean_barcode_text(data)  # cleaned data
        })

    return barcodes


# =============================
# TEXT CLEANING
# =============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s./()\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# keep only numbers
def digits_only(text):
    return re.sub(r'[^0-9]', '', str(text))


# =============================
# LOAD CSV FILE
# =============================

def load_items(csv_file):

    # try different encodings (important!)
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']

    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_file, encoding=enc, encoding_errors='ignore')
            break
        except:
            pass

    if df is None:
        raise ValueError("Cannot read CSV")

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # detect columns dynamically
    code_col = next((c for c in ['itemcode'] if c in df.columns), None)
    desc_col = next((c for c in ['description'] if c in df.columns), None)
    gtin_col = next((c for c in ['gtin'] if c in df.columns), None)

    # clean data
    df[code_col] = df[code_col].astype(str).str.strip()

    if gtin_col:
        df[gtin_col] = df[gtin_col].astype(str).apply(digits_only)

    return df, code_col, desc_col, gtin_col


# =============================
# DETECT DATA FROM TEXT
# =============================

# detect GTIN
def detect_gtin(raw_text):
    text = re.sub(r'[^0-9]', ' ', str(raw_text))
    candidates = re.findall(r'\d{14}|\d{13}|\d{12}|\d{8}', text)
    return candidates[0] if candidates else ''


# detect item code
def detect_code_from_text(raw_text):
    text = str(raw_text).upper()

    matches = re.findall(r'\b[A-Z]{2,10}\d{2,10}\b', text)

    return matches[0] if matches else ''


# detect batch
def detect_batch(raw_text):
    text = str(raw_text).upper()

    m = re.search(r'(LOT|BATCH)[\s:]*([A-Z0-9]+)', text)
    if m:
        return m.group(2)

    return ''


# =============================
# MATCHING LOGIC
# =============================

# match by GTIN
def find_exact_gtin_match(gtin, df, gtin_col):
    if not gtin or not gtin_col:
        return None
    return df[df[gtin_col] == gtin].head(1)


# match by item code
def find_exact_code_match(code, df, code_col):
    if not code:
        return None
    return df[df[code_col].str.upper() == code.upper()].head(1)


# fuzzy match
def find_best_item_match(text, df, code_col, desc_col):
    best_score = 0
    best_row = None

    for _, row in df.iterrows():
        score = fuzz.partial_ratio(text, str(row[code_col]))

        if score > best_score:
            best_score = score
            best_row = row

    return best_row, best_score


# =============================
# MAIN EXTRACTION FUNCTION
# =============================

def extract_all_data(file_bytes):

    # OCR text
    raw_text, lines = extract_text_from_bytes(file_bytes)

    # barcode
    barcodes = decode_barcodes_from_bytes(file_bytes)

    # detect values
    gtin = detect_gtin(raw_text)
    code = detect_code_from_text(raw_text)
    batch = detect_batch(raw_text)

    return {
        "ocr_text": raw_text,
        "barcodes": barcodes,
        "detected_gtin": gtin,
        "detected_code": code,
        "detected_batch": batch
    }


# =============================
# ROUTES (API)
# =============================

@app.route("/")
def home():
    return "OCR + Barcode + Item Match API is running"


# OCR only
@app.route("/ocr", methods=["POST"])
def ocr():
    file = request.files["file"]
    data = extract_all_data(file.read())
    return jsonify(data)


# match item
@app.route("/match-item", methods=["POST"])
def match_item():

    file = request.files["file"]
    file_bytes = file.read()

    data = extract_all_data(file_bytes)

    df, code_col, desc_col, gtin_col = load_items(CSV_FILE_PATH)

    # 1. GTIN match
    match = find_exact_gtin_match(data["detected_gtin"], df, gtin_col)
    if match is not None and not match.empty:
        return jsonify({"match": "GTIN", "data": match.iloc[0].to_dict()})

    # 2. Code match
    match = find_exact_code_match(data["detected_code"], df, code_col)
    if match is not None and not match.empty:
        return jsonify({"match": "CODE", "data": match.iloc[0].to_dict()})

    # 3. Fuzzy
    best_row, score = find_best_item_match(data["ocr_text"], df, code_col, desc_col)

    return jsonify({
        "match": "FUZZY",
        "score": score,
        "data": best_row.to_dict() if best_row is not None else {}
    })


# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
