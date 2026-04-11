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
from pyzbar.pyzbar import decode as zbar_decode

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
def open_image_from_bytes(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

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
# BARCODE DETECTION
# ----------------------------
def clean_barcode_text(s):
    if not s:
        return ""
    s = str(s).strip()
    s = s.replace("\n", "")
    s = s.replace("\r", "")
    s = re.sub(r"\s+", "", s)
    return s

def decode_barcodes_from_bytes(file_bytes):
    img = open_image_from_bytes(file_bytes)
    decoded = zbar_decode(img)

    barcodes = []
    for obj in decoded:
        try:
            data = obj.data.decode("utf-8", errors="ignore").strip()
        except Exception:
            data = str(obj.data)

        barcodes.append({
            "type": str(obj.type),
            "data": clean_barcode_text(data)
        })

    return barcodes

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s./()\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def digits_only(text):
    return re.sub(r'[^0-9]', '', str(text))

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

    possible_code_cols = ['item_code', 'itemcode', 'item code', 'code', 'sku', 'itemid']
    possible_desc_cols = ['description', 'item_name', 'item name', 'name', 'item', 'displayname']
    possible_gtin_cols = ['gtin', 'barcode', 'ean', 'upc', 'gs1']

    code_col = next((c for c in possible_code_cols if c in df.columns), None)
    desc_col = next((c for c in possible_desc_cols if c in df.columns), None)
    gtin_col = next((c for c in possible_gtin_cols if c in df.columns), None)

    if not code_col:
        raise ValueError(f"No item code column found. CSV columns: {list(df.columns)}")

    df[code_col] = df[code_col].fillna('').astype(str).str.strip()
    if desc_col:
        df[desc_col] = df[desc_col].fillna('').astype(str).str.strip()
    if gtin_col:
        df[gtin_col] = df[gtin_col].fillna('').astype(str).apply(digits_only)

    if desc_col:
        df['combined'] = (
            df[code_col].fillna('').astype(str) + ' ' +
            df[desc_col].fillna('').astype(str)
        )
    else:
        df['combined'] = df[code_col].fillna('').astype(str)

    df['combined'] = df['combined'].apply(clean_text)

    return df, code_col, desc_col, gtin_col

# ----------------------------
# GS1 PARSING
# ----------------------------
def parse_gs1_text(raw):
    """
    Basic GS1 parser for common AIs:
    (01) GTIN
    (17) Expiry YYMMDD
    (10) Batch/Lot
    (20) Variant/secondary code
    """
    result = {
        "gtin": "",
        "batch": "",
        "expiry_date": "",
        "serial_or_variant": ""
    }

    if not raw:
        return result

    raw = clean_barcode_text(raw)

    # Already bracketed format
    m = re.search(r'\(01\)(\d{14})', raw)
    if m:
        result["gtin"] = m.group(1)

    m = re.search(r'\(17\)(\d{6})', raw)
    if m:
        result["expiry_date"] = format_yy_mm_dd(m.group(1))

    m = re.search(r'\(10\)([A-Za-z0-9\-/\.]+)', raw)
    if m:
        result["batch"] = m.group(1)

    m = re.search(r'\(20\)([A-Za-z0-9\-/\.]+)', raw)
    if m:
        result["serial_or_variant"] = m.group(1)

    # Fallback if barcode is plain numeric GTIN only
    if not result["gtin"]:
        digits = digits_only(raw)
        if len(digits) in (8, 12, 13, 14):
            result["gtin"] = digits

    return result

def format_yy_mm_dd(v):
    if not v or len(v) != 6:
        return ""
    yy = int(v[:2])
    mm = v[2:4]
    dd = v[4:6]
    year = 2000 + yy
    return f"{year:04d}-{mm}-{dd}"

def extract_barcode_fields(barcodes):
    """
    Prefer GS1 barcode values if found.
    """
    output = {
        "barcode_texts": [],
        "barcode_gtin": "",
        "barcode_batch": "",
        "barcode_expiry_date": "",
        "barcode_serial_or_variant": ""
    }

    for b in barcodes:
        data = clean_barcode_text(b.get("data", ""))
        if not data:
            continue

        output["barcode_texts"].append(data)

        parsed = parse_gs1_text(data)

        if not output["barcode_gtin"] and parsed["gtin"]:
            output["barcode_gtin"] = parsed["gtin"]

        if not output["barcode_batch"] and parsed["batch"]:
            output["barcode_batch"] = parsed["batch"]

        if not output["barcode_expiry_date"] and parsed["expiry_date"]:
            output["barcode_expiry_date"] = parsed["expiry_date"]

        if not output["barcode_serial_or_variant"] and parsed["serial_or_variant"]:
            output["barcode_serial_or_variant"] = parsed["serial_or_variant"]

    return output

# ----------------------------
# DETECT GTIN
# ----------------------------
def detect_gtin(raw_text):
    text = re.sub(r'[^0-9]', ' ', str(raw_text))
    candidates = re.findall(r'\b\d{14}\b|\b\d{13}\b|\b\d{12}\b|\b\d{8}\b', text)
    return candidates[0] if candidates else ''

# ----------------------------
# DETECT EXPIRY DATE
# ----------------------------
def detect_expiry_date(raw_text):
    text = str(raw_text)

    # GS1 AI (17) YYMMDD
    m = re.search(r'\(17\)\s*(\d{6})', text)
    if m:
        return format_yy_mm_dd(m.group(1))

    patterns = [
        r'\b(20\d{2}-\d{2}-\d{2})\b',
        r'\b(20\d{2}/\d{2}/\d{2})\b',
        r'\b(0[1-9]|1[0-2])[\/\-](20\d{2})\b',
        r'\b(0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-](20\d{2})\b'
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(0)

    return ''

# ----------------------------
# DETECT ITEM CODE
# ----------------------------
def detect_code_from_text(raw_text):
    text = str(raw_text).upper()
    gtin = detect_gtin(text)
    expiry = detect_expiry_date(text)

    # Prefer alphanumeric codes like GIA6038S / IPC0026 / DSBI919FX
    alpha_num_patterns = [
        r'\b([A-Z]{2,15}\d{2,15}[A-Z0-9]*)\b',
        r'\b([A-Z]{1,10}-\d{2,15}[A-Z0-9\-]*)\b',
        r'\b([A-Z]{2,15}\.\d{2,15}[A-Z0-9\.]*)\b'
    ]

    for pattern in alpha_num_patterns:
        matches = re.findall(pattern, text)
        for value in matches:
            if value != gtin and value not in ('LOT', 'BATCH', 'USEBY'):
                return value

    # Fallback numeric code
    numeric_candidates = re.findall(r'\b\d{5,10}\b', text)
    for num in numeric_candidates:
        if num == gtin:
            continue
        if expiry and num in expiry:
            continue
        if num.startswith('20'):
            continue
        if num in ('300731', '300831', '01', '10', '17', '20'):
            continue
        return num

    return ''

# ----------------------------
# DETECT BATCH
# ----------------------------
def detect_batch(raw_text):
    text = str(raw_text).upper()

    labeled_patterns = [
        r'\b(?:LOT|BATCH|LOT NO|LOT NUMBER|BN|LN)[\s:.\-]*([A-Z0-9\-\/]{4,30})\b',
        r'\(10\)\s*([A-Z0-9\-\/]{4,30})\b'
    ]

    for pattern in labeled_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    detected_code = detect_code_from_text(text)
    detected_gtin = detect_gtin(text)
    detected_expiry = detect_expiry_date(text)

    tokens = re.findall(r'\b[A-Z0-9.\-\/]{4,30}\b', text)

    skip_words = {
        'COVIDIEN', 'SOFSILK', 'RELIAPOINT', 'CUTTING', 'BLACK',
        'METRIC', 'BRAIDED', 'COATED', 'STAINLESS', 'STEEL',
        'RAMEEM', 'MEDICA', 'ARABIA', 'KINGDOM', 'SAUDI',
        'LOT', 'USE', 'BY'
    }

    for token in tokens:
        if token in skip_words:
            continue
        if token == detected_code or token == detected_gtin or token == detected_expiry:
            continue

        if re.fullmatch(r'\d{8,14}', token):
            continue
        if re.fullmatch(r'20\d{2}-\d{2}-\d{2}', token):
            continue
        if re.fullmatch(r'\d+(\.\d+)?(MM|CM|M)?', token):
            continue
        if re.fullmatch(r'\d+/\d+', token):
            continue

        if len(token) >= 5 and re.search(r'[A-Z]', token) and re.search(r'\d', token):
            return token

    return ''

# ----------------------------
# MATCHING
# ----------------------------
def find_exact_code_match(detected_code, df, code_col):
    if not detected_code:
        return None
    matches = df[df[code_col].astype(str).str.upper() == str(detected_code).upper()]
    if not matches.empty:
        return matches.iloc[0]
    return None

def find_exact_gtin_match(detected_gtin, df, gtin_col):
    if not detected_gtin or not gtin_col:
        return None
    detected_gtin = digits_only(detected_gtin)
    matches = df[df[gtin_col].astype(str) == detected_gtin]
    if not matches.empty:
        return matches.iloc[0]
    return None

def find_best_item_match(raw_text, df, code_col, desc_col):
    text_clean = clean_text(raw_text)
    best_score = 0
    best_row = None
    best_type = "none"

    for _, row in df.iterrows():
        code_value = str(row[code_col]) if pd.notna(row[code_col]) else ""
        desc_value = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else ""

        code_clean = clean_text(code_value)
        desc_clean = clean_text(desc_value)
        combined_clean = clean_text(code_value + " " + desc_value)

        score_code = fuzz.partial_ratio(text_clean, code_clean) if code_clean else 0
        score_desc = fuzz.partial_ratio(text_clean, desc_clean) if desc_clean else 0
        score_combined = fuzz.partial_ratio(text_clean, combined_clean) if combined_clean else 0

        row_best_score = max(score_code, score_desc, score_combined)

        if row_best_score > best_score:
            best_score = row_best_score
            best_row = row
            if row_best_score == score_code:
                best_type = "code"
            elif row_best_score == score_desc:
                best_type = "description"
            else:
                best_type = "combined"

    return best_row, best_score, best_type

# ----------------------------
# UNIFIED EXTRACTION
# ----------------------------
def extract_all_data(file_bytes):
    raw_text, lines = extract_text_from_bytes(file_bytes)
    barcodes = decode_barcodes_from_bytes(file_bytes)
    barcode_fields = extract_barcode_fields(barcodes)

    ocr_gtin = detect_gtin(raw_text)
    ocr_code = detect_code_from_text(raw_text)
    ocr_batch = detect_batch(raw_text)
    ocr_expiry = detect_expiry_date(raw_text)

    final_gtin = barcode_fields["barcode_gtin"] or ocr_gtin
    final_batch = barcode_fields["barcode_batch"] or ocr_batch
    final_expiry = barcode_fields["barcode_expiry_date"] or ocr_expiry
    final_code = ocr_code

    return {
        "ocr_text": raw_text,
        "lines": lines,
        "barcodes": barcodes,
        "barcode_texts": barcode_fields["barcode_texts"],
        "barcode_gtin": barcode_fields["barcode_gtin"],
        "barcode_batch": barcode_fields["barcode_batch"],
        "barcode_expiry_date": barcode_fields["barcode_expiry_date"],
        "barcode_serial_or_variant": barcode_fields["barcode_serial_or_variant"],
        "ocr_gtin": ocr_gtin,
        "ocr_detected_code": ocr_code,
        "ocr_batch": ocr_batch,
        "ocr_expiry_date": ocr_expiry,
        "detected_gtin": final_gtin,
        "detected_code": final_code,
        "detected_batch": final_batch,
        "detected_expiry_date": final_expiry
    }

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return "OCR + Barcode + Item Match API is running"

@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({"success": False, "error": "Empty file"}), 400

        extracted = extract_all_data(file_bytes)

        return jsonify({
            "success": True,
            **extracted
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/match-item", methods=["POST"])
def match_item():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        if not os.path.exists(CSV_FILE_PATH):
            return jsonify({
                "success": False,
                "error": f"CSV file not found on server: {CSV_FILE_PATH}"
            }), 500

        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({"success": False, "error": "Empty file"}), 400

        extracted = extract_all_data(file_bytes)

        detected_gtin = extracted["detected_gtin"]
        detected_code = extracted["detected_code"]
        detected_batch = extracted["detected_batch"]
        detected_expiry = extracted["detected_expiry_date"]

        df, code_col, desc_col, gtin_col = load_items(CSV_FILE_PATH)

        MIN_MATCH_SCORE = 70

        # 1) GTIN exact match
        gtin_row = find_exact_gtin_match(detected_gtin, df, gtin_col)
        if gtin_row is not None:
            return jsonify({
                "success": True,
                **extracted,
                "matched_item_code": str(gtin_row[code_col]),
                "matched_description": str(gtin_row[desc_col]) if desc_col else "",
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin,
                "match_score": 100.0,
                "match_type": "gtin_exact",
                "verification_order": "gtin_first"
            })

        # 2) item code exact match
        exact_row = find_exact_code_match(detected_code, df, code_col)
        if exact_row is not None:
            gtin_value = ""
            if gtin_col and gtin_col in exact_row:
                gtin_value = str(exact_row[gtin_col])

            return jsonify({
                "success": True,
                **extracted,
                "matched_item_code": str(exact_row[code_col]),
                "matched_description": str(exact_row[desc_col]) if desc_col else "",
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin or gtin_value,
                "match_score": 100.0,
                "match_type": "exact_code",
                "verification_order": "gtin_first_then_item_code"
            })

        # 3) fuzzy compare
        fuzzy_source_text = " ".join([
            extracted.get("ocr_text", ""),
            extracted.get("detected_code", ""),
            extracted.get("detected_gtin", "")
        ]).strip()

        best_row, best_score, best_match_basis = find_best_item_match(
            fuzzy_source_text, df, code_col, desc_col
        )

        if best_row is None or float(best_score) < MIN_MATCH_SCORE:
            return jsonify({
                "success": True,
                **extracted,
                "matched_item_code": "UNKNOWN PRODUCT",
                "matched_description": "UNKNOWN PRODUCT",
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin,
                "match_score": round(float(best_score), 2) if best_row is not None else 0,
                "match_type": "unknown_product",
                "verification_order": "gtin_first_then_item_code"
            })

        matched_item_code = str(best_row[code_col])
        matched_description = str(best_row[desc_col]) if desc_col else ""
        matched_gtin = ""
        if gtin_col and gtin_col in best_row:
            matched_gtin = str(best_row[gtin_col])

        return jsonify({
            "success": True,
            **extracted,
            "matched_item_code": matched_item_code,
            "matched_description": matched_description,
            "batch": detected_batch,
            "expiry_date": detected_expiry,
            "gtin": detected_gtin or matched_gtin,
            "match_score": round(float(best_score), 2),
            "match_type": f"fuzzy_{best_match_basis}",
            "verification_order": "gtin_first_then_item_code"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
