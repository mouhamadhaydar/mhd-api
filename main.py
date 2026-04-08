from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import easyocr
import io
import os
import torch
import numpy as np
import re
import pandas as pd
from rapidfuzz import fuzz

app = Flask(__name__)

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("EASYOCR_MODULE_PATH", os.path.join(BASE_DIR, ".EasyOCR"))
CSV_FILE_PATH = os.path.join(BASE_DIR, "items.csv")

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=True
)

# ----------------------------
# IMAGE PROCESSING
# ----------------------------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = ImageOps.autocontrast(img)
    return np.array(img)

def extract_text_from_bytes(file_bytes):
    img = preprocess_image(file_bytes)
    results = reader.readtext(img, detail=0, paragraph=True)
    full_text = " ".join(results).strip()
    return full_text, results

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s./-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# LOAD ITEMS CSV
# ----------------------------
def load_items(csv_file):
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']

    last_error = None
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_file, encoding=enc, encoding_errors='ignore')
            break
        except Exception as e:
            last_error = e

    if df is None:
        raise ValueError(f"Could not read CSV file with supported encodings. Last error: {last_error}")

    df.columns = df.columns.str.strip().str.lower()

    possible_code_cols = ['item_code', 'itemcode', 'code', 'sku']
    possible_desc_cols = ['description', 'item_name', 'name', 'item', 'displayname']

    code_col = next((c for c in possible_code_cols if c in df.columns), None)
    desc_col = next((c for c in possible_desc_cols if c in df.columns), None)

    if not code_col:
        raise ValueError(f"No item code column found. CSV columns: {list(df.columns)}")

    if desc_col:
        df['combined'] = (
            df[code_col].fillna('').astype(str) + ' ' +
            df[desc_col].fillna('').astype(str)
        )
    else:
        df['combined'] = df[code_col].fillna('').astype(str)

    df['combined'] = df['combined'].apply(clean_text)

    return df, code_col, desc_col

# ----------------------------
# DETECT ITEM CODE
# ----------------------------
def detect_code_from_text(raw_text):
    text = str(raw_text).upper()

    patterns = [
        r'\b([A-Z]{2,10}\.\d{2,10})\b',   # RML.12058
        r'\b([A-Z]{1,5}-\d{2,10})\b',     # SS-683
        r'\b([A-Z]{2,10}\d{2,10})\b'      # IPC00025
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    return ''

# ----------------------------
# DETECT GTIN
# ----------------------------
def detect_gtin(raw_text):
    text = re.sub(r'[^0-9]', ' ', str(raw_text))
    candidates = re.findall(r'\b\d{8}\b|\b\d{12}\b|\b\d{13}\b|\b\d{14}\b', text)
    return candidates[0] if candidates else ''

# ----------------------------
# DETECT EXPIRY DATE
# ----------------------------
def detect_expiry_date(raw_text):
    text = str(raw_text)

    patterns = [
        r'\b(20\d{2}-\d{2}-\d{2})\b',
        r'\b(0[1-9]|1[0-2])[\/\-](20\d{2})\b',
        r'\b(0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-](20\d{2})\b'
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(0)

    return ''

# ----------------------------
# DETECT BATCH (FINAL FIXED)
# ----------------------------
def detect_batch(raw_text):
    text = str(raw_text).upper()

    labeled_patterns = [
        r'\b(?:BATCH|LOT|LOT NO|LOT NUMBER|BN|LN)[\s:.\-]*([A-Z0-9\-\/]{4,30})\b'
    ]

    for pattern in labeled_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    detected_code = detect_code_from_text(text)
    detected_gtin = detect_gtin(text)
    detected_expiry = detect_expiry_date(text)

    tokens = re.findall(r'\b[A-Z0-9.\-\/]{5,20}\b', text)

    skip_words = {
        'COVIDIEN', 'SOFSILK', 'RELIAPOINT', 'CUTTING', 'BLACK',
        'METRIC', 'BRAIDED', 'COATED', 'STAINLESS', 'STEEL',
        'RAMEEM', 'MEDICA', 'ARABIA', 'KINGDOM', 'SAUDI'
    }

    skip_exact_tokens = {'C-13'}

    for token in tokens:
        if token in skip_words or token in skip_exact_tokens:
            continue
        if token == detected_code:
            continue
        if token == detected_gtin:
            continue
        if token == detected_expiry:
            continue

        if re.fullmatch(r'\d{8,14}', token):
            continue
        if re.fullmatch(r'20\d{2}-\d{2}-\d{2}', token):
            continue
        if re.fullmatch(r'[A-Z]{1,5}-\d{2,10}', token):
            continue
        if re.fullmatch(r'[A-Z]{2,10}\.\d{2,10}', token):
            continue
        if re.fullmatch(r'[A-Z]{2,10}\d{2,10}', token):
            continue
        if re.fullmatch(r'\d+(\.\d+)?(MM|CM|M)?', token):
            continue
        if re.fullmatch(r'\d+/\d+', token):
            continue
        if re.fullmatch(r'[A-Z]-\d{1,3}', token):
            continue

        if len(token) >= 6 and re.search(r'[A-Z]', token) and re.search(r'\d', token):
            return token

    return ''

# ----------------------------
# MATCHING
# ----------------------------
def find_exact_code_match(detected_code, df, code_col):
    if not detected_code:
        return None

    matches = df[df[code_col].astype(str).str.upper() == detected_code.upper()]
    if not matches.empty:
        return matches.iloc[0]

    return None

def find_best_match(ocr_text, df):
    best_score = 0
    best_row = None

    for _, row in df.iterrows():
        score = fuzz.partial_ratio(ocr_text, row['combined'])
        if score > best_score:
            best_score = score
            best_row = row

    return best_row, best_score

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return "OCR + Item Match API is running 🚀"

@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        raw_text, lines = extract_text_from_bytes(file_bytes)

        return jsonify({
            "success": True,
            "text": raw_text,
            "lines": lines,
            "detected_code": detect_code_from_text(raw_text),
            "batch": detect_batch(raw_text),
            "expiry_date": detect_expiry_date(raw_text),
            "gtin": detect_gtin(raw_text)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/match-item", methods=["POST"])
def match_item():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        raw_text, lines = extract_text_from_bytes(file_bytes)
        cleaned_text = clean_text(raw_text)

        detected_code = detect_code_from_text(raw_text)
        detected_batch = detect_batch(raw_text)
        detected_expiry = detect_expiry_date(raw_text)
        detected_gtin = detect_gtin(raw_text)

        df, code_col, desc_col = load_items(CSV_FILE_PATH)

        exact_row = find_exact_code_match(detected_code, df, code_col)

        if exact_row is not None:
            return jsonify({
                "success": True,
                "ocr_text": raw_text,
                "detected_code": detected_code,
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin,
                "matched_item_code": str(exact_row[code_col]),
                "matched_description": str(exact_row[desc_col]) if desc_col else "",
                "match_score": 100,
                "match_type": "exact"
            })

        best_row, best_score = find_best_match(cleaned_text, df)

        return jsonify({
            "success": True,
            "ocr_text": raw_text,
            "detected_code": detected_code,
            "batch": detected_batch,
            "expiry_date": detected_expiry,
            "gtin": detected_gtin,
            "matched_item_code": str(best_row[code_col]) if best_row is not None else "",
            "matched_description": str(best_row[desc_col]) if desc_col and best_row is not None else "",
            "match_score": round(float(best_score), 2),
            "match_type": "fuzzy"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
