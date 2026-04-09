from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageFilter
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

# =========================================================
# IMAGE PROCESSING
# =========================================================
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    # light threshold to improve printed labels
    img = img.point(lambda p: 255 if p > 160 else 0)

    return np.array(img)


def extract_text_from_bytes(file_bytes):
    img = preprocess_image(file_bytes)

    results = reader.readtext(
        img,
        detail=0,
        paragraph=False,
        decoder='greedy'
    )

    cleaned_lines = []
    for r in results:
        line = str(r).strip()
        if line:
            cleaned_lines.append(line)

    full_text = " ".join(cleaned_lines).strip()
    return full_text, cleaned_lines


# =========================================================
# CLEAN TEXT
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s./()\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================================================
# OCR CONFUSION FIXER
# =========================================================
def fix_ocr_confusions(token, field_type="generic"):
    token = str(token).strip().upper()
    token = re.sub(r'\s+', '', token)

    if not token:
        return token

    num_to_alpha = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '8': 'B',
        '2': 'Z'
    }

    alpha_to_num = {
        'O': '0',
        'Q': '0',
        'D': '0',
        'I': '1',
        'L': '1',
        '|': '1',
        'S': '5',
        'B': '8',
        'Z': '2'
    }

    chars = list(token)

    if field_type == "gtin":
        fixed = []
        for c in chars:
            fixed.append(alpha_to_num.get(c, c))
        return ''.join(fixed)

    if field_type == "expiry":
        fixed = []
        for c in chars:
            if c in '/-.':
                fixed.append(c)
            else:
                fixed.append(alpha_to_num.get(c, c))
        return ''.join(fixed)

    if field_type == "code":
        # item codes can be numeric or alphanumeric
        # do not aggressively convert
        return token

    if field_type == "batch":
        fixed = chars[:]

        for i, c in enumerate(fixed):
            prev_c = fixed[i - 1] if i > 0 else ''
            next_c = fixed[i + 1] if i < len(fixed) - 1 else ''

            prev_is_digit = prev_c.isdigit()
            next_is_digit = next_c.isdigit()
            prev_is_alpha = prev_c.isalpha()
            next_is_alpha = next_c.isalpha()

            # convert numeric-looking chars to letters if batch looks alphanumeric
            if c == '8':
                if prev_is_digit and (next_is_alpha or next_c == '' or prev_is_alpha):
                    fixed[i] = 'B'
            elif c == '0':
                if prev_is_alpha and not next_is_digit:
                    fixed[i] = 'O'
            elif c == '1':
                if prev_is_alpha and not next_is_digit:
                    fixed[i] = 'I'
            elif c in ('O', 'I', 'L', 'S', 'B', 'Z'):
                if prev_is_digit or next_is_digit:
                    fixed[i] = alpha_to_num.get(c, c)

        return ''.join(fixed)

    return token


# =========================================================
# LOAD ITEMS CSV
# =========================================================
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

    possible_code_cols = ['item_code', 'itemcode', 'code', 'sku', 'itemid']
    possible_desc_cols = ['description', 'item_name', 'name', 'item', 'displayname']
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
        df[gtin_col] = (
            df[gtin_col]
            .fillna('')
            .astype(str)
            .apply(lambda x: fix_ocr_confusions(x, "gtin"))
            .str.replace(r'[^0-9]', '', regex=True)
        )

    if desc_col:
        df['combined'] = (
            df[code_col].fillna('').astype(str) + ' ' +
            df[desc_col].fillna('').astype(str)
        )
    else:
        df['combined'] = df[code_col].fillna('').astype(str)

    df['combined'] = df['combined'].apply(clean_text)

    return df, code_col, desc_col, gtin_col


# =========================================================
# DETECT GTIN
# =========================================================
def detect_gtin(raw_text):
    text = str(raw_text).upper()

    # Try GS1 AI (01) first
    m = re.search(r'\(01\)\s*([A-Z0-9]{8,18})', text)
    if m:
        gtin = fix_ocr_confusions(m.group(1), "gtin")
        gtin = re.sub(r'[^0-9]', '', gtin)
        if len(gtin) in (8, 12, 13, 14):
            return gtin

    # fallback general token scan
    cleaned = re.sub(r'[^A-Z0-9]', ' ', text)
    tokens = cleaned.split()

    for token in tokens:
        fixed = fix_ocr_confusions(token, "gtin")
        fixed = re.sub(r'[^0-9]', '', fixed)
        if len(fixed) in (8, 12, 13, 14):
            return fixed

    return ''


# =========================================================
# DETECT EXPIRY DATE
# =========================================================
def detect_expiry_date(raw_text):
    text = str(raw_text).upper()

    labeled_patterns = [
        r'\bEXP(?:IRY)?[\s:.-]*(20\d{2}[\/\-](0[1-9]|1[0-2]))\b',
        r'\bEXP(?:IRY)?[\s:.-]*((0[1-9]|1[0-2])[\/\-]20\d{2})\b',
        r'\bEXP(?:IRY)?[\s:.-]*(20\d{2}[\/\-](0[1-9]|1[0-2])[\/\-](0[1-9]|[12][0-9]|3[01]))\b',
        r'\bEXP(?:IRY)?[\s:.-]*((0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-]20\d{2})\b'
    ]

    for pattern in labeled_patterns:
        m = re.search(pattern, text)
        if m:
            return fix_ocr_confusions(m.group(1), "expiry")

    generic_patterns = [
        r'\b(20\d{2}[\/\-](0[1-9]|1[0-2]))\b',
        r'\b((0[1-9]|1[0-2])[\/\-]20\d{2})\b',
        r'\b(20\d{2}[\/\-](0[1-9]|1[0-2])[\/\-](0[1-9]|[12][0-9]|3[01]))\b',
        r'\b((0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-]20\d{2})\b'
    ]

    for pattern in generic_patterns:
        m = re.search(pattern, text)
        if m:
            return fix_ocr_confusions(m.group(1), "expiry")

    return ''


# =========================================================
# DETECT ITEM CODE
# =========================================================
def detect_code_from_text(raw_text):
    text = str(raw_text).upper()

    # 1) Prefer explicit SKU / ITEM / CODE labels
    labeled_patterns = [
        r'\bSKU[\s:.-]*([A-Z0-9.\-]{3,30})\b',
        r'\bITEM[\s:.-]*([A-Z0-9.\-]{3,30})\b',
        r'\bITEM\s*CODE[\s:.-]*([A-Z0-9.\-]{3,30})\b',
        r'\bCODE[\s:.-]*([A-Z0-9.\-]{3,30})\b'
    ]
    for pattern in labeled_patterns:
        m = re.search(pattern, text)
        if m:
            return fix_ocr_confusions(m.group(1), "code")

    gtin = detect_gtin(text)
    expiry = detect_expiry_date(text)

    # 2) standalone numeric candidates
    numeric_candidates = re.findall(r'\b\d{5,10}\b', text)
    skip_values = {'01', '10', '11', '17', '30'}

    for num in numeric_candidates:
        if num == gtin:
            continue
        if expiry and num in expiry:
            continue
        if num.startswith('20'):
            continue
        if num in skip_values:
            continue
        return fix_ocr_confusions(num, "code")

    # 3) alphanumeric candidates
    patterns = [
        r'\b([A-Z]{2,10}\.\d{2,10})\b',
        r'\b([A-Z]{1,10}-\d{2,10})\b',
        r'\b([A-Z]{1,10}\d{2,20})\b',
        r'\b(\d{2,20}[A-Z]{1,10})\b'
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = m.group(1)
            if value != gtin:
                return fix_ocr_confusions(value, "code")

    return ''


# =========================================================
# DETECT BATCH
# =========================================================
def detect_batch(raw_text):
    text = str(raw_text).upper()

    labeled_patterns = [
        r'\b(?:LOT|BATCH|LOT NO|LOT NUMBER|BN|LN)[\s:.\-]*([A-Z0-9\-\/]{3,30})\b',
        r'\(10\)\s*([A-Z0-9\-\/]{3,30})\b'
    ]

    for pattern in labeled_patterns:
        m = re.search(pattern, text)
        if m:
            return fix_ocr_confusions(m.group(1), "batch")

    tokens = re.findall(r'\b[A-Z0-9.\-\/]{4,30}\b', text)

    detected_code = detect_code_from_text(text)
    detected_gtin = detect_gtin(text)
    detected_expiry = detect_expiry_date(text)

    skip_words = {
        'LOT', 'BATCH', 'EXP', 'MFG', 'SKU', 'QTY', 'NDC',
        'COVIDIEN', 'SOFSILK', 'RELIAPOINT', 'STAINLESS', 'STEEL',
        'BOTTLES', 'VIAL', 'VIALS', 'ML'
    }

    for token in tokens:
        token_fixed = fix_ocr_confusions(token, "batch")

        if token_fixed in skip_words:
            continue
        if token_fixed == detected_code or token_fixed == detected_gtin or token_fixed == detected_expiry:
            continue
        if re.fullmatch(r'\d{8,14}', token_fixed):
            continue
        if re.fullmatch(r'20\d{2}[\/\-](0[1-9]|1[0-2])', token_fixed):
            continue
        if re.fullmatch(r'(0[1-9]|1[0-2])[\/\-]20\d{2}', token_fixed):
            continue

        if len(token_fixed) >= 4 and re.search(r'[A-Z]', token_fixed) and re.search(r'\d', token_fixed):
            return token_fixed

    return ''


# =========================================================
# VALIDATORS
# =========================================================
def validate_batch(batch):
    if not batch:
        return ''
    batch = str(batch).strip().upper()
    if re.fullmatch(r'[A-Z0-9\-\/]{3,30}', batch):
        return batch
    return ''


def validate_expiry(expiry):
    if not expiry:
        return ''
    expiry = str(expiry).strip()
    patterns = [
        r'20\d{2}[\/\-](0[1-9]|1[0-2])',
        r'(0[1-9]|1[0-2])[\/\-]20\d{2}',
        r'20\d{2}[\/\-](0[1-9]|1[0-2])[\/\-](0[1-9]|[12][0-9]|3[01])',
        r'(0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-]20\d{2}'
    ]
    for pattern in patterns:
        if re.fullmatch(pattern, expiry):
            return expiry
    return ''


# =========================================================
# MATCHING
# =========================================================
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
    matches = df[df[gtin_col].astype(str) == str(detected_gtin)]
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


# =========================================================
# BUILD RESPONSE
# =========================================================
def build_response_payload(raw_text, lines, detected_code, detected_batch, detected_expiry, detected_gtin,
                           matched_item_code="", matched_description="", match_score=0, match_type="none"):
    return {
        "success": True,
        "ocr_text": raw_text,
        "lines": lines,
        "detected_code": detected_code,
        "matched_item_code": matched_item_code,
        "matched_description": matched_description,
        "batch": detected_batch,
        "expiry_date": detected_expiry,
        "gtin": detected_gtin,
        "match_score": round(float(match_score), 2) if match_score is not None else 0,
        "match_type": match_type
    }


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return "OCR + Item Match API is running"


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({"success": False, "error": "Empty file"}), 400

        raw_text, lines = extract_text_from_bytes(file_bytes)

        detected_code = detect_code_from_text(raw_text)
        detected_batch = validate_batch(detect_batch(raw_text))
        detected_expiry = validate_expiry(detect_expiry_date(raw_text))
        detected_gtin = detect_gtin(raw_text)

        return jsonify({
            "success": True,
            "text": raw_text,
            "lines": lines,
            "detected_code": detected_code,
            "batch": detected_batch,
            "expiry_date": detected_expiry,
            "gtin": detected_gtin
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

        raw_text, lines = extract_text_from_bytes(file_bytes)

        detected_code = detect_code_from_text(raw_text)
        detected_batch = validate_batch(detect_batch(raw_text))
        detected_expiry = validate_expiry(detect_expiry_date(raw_text))
        detected_gtin = detect_gtin(raw_text)

        df, code_col, desc_col, gtin_col = load_items(CSV_FILE_PATH)

        MIN_MATCH_SCORE = 70

        # 1) GTIN exact match
        gtin_row = find_exact_gtin_match(detected_gtin, df, gtin_col)
        if gtin_row is not None:
            return jsonify(build_response_payload(
                raw_text=raw_text,
                lines=lines,
                detected_code=detected_code,
                detected_batch=detected_batch,
                detected_expiry=detected_expiry,
                detected_gtin=detected_gtin,
                matched_item_code=str(gtin_row[code_col]),
                matched_description=str(gtin_row[desc_col]) if desc_col else "",
                match_score=100.0,
                match_type="gtin_exact"
            ))

        # 2) Item code exact match
        exact_row = find_exact_code_match(detected_code, df, code_col)
        if exact_row is not None:
            return jsonify(build_response_payload(
                raw_text=raw_text,
                lines=lines,
                detected_code=detected_code,
                detected_batch=detected_batch,
                detected_expiry=detected_expiry,
                detected_gtin=detected_gtin,
                matched_item_code=str(exact_row[code_col]),
                matched_description=str(exact_row[desc_col]) if desc_col else "",
                match_score=100.0,
                match_type="exact_code"
            ))

        # 3) fuzzy match
        best_row, best_score, best_match_basis = find_best_item_match(raw_text, df, code_col, desc_col)

        # 4) unknown if low score
        if best_row is None or float(best_score) < MIN_MATCH_SCORE:
            return jsonify(build_response_payload(
                raw_text=raw_text,
                lines=lines,
                detected_code=detected_code,
                detected_batch=detected_batch,
                detected_expiry=detected_expiry,
                detected_gtin=detected_gtin,
                matched_item_code="UNKNOWN PRODUCT",
                matched_description="UNKNOWN PRODUCT",
                match_score=round(float(best_score), 2) if best_row is not None else 0,
                match_type="unknown_product"
            ))

        # 5) valid fuzzy match
        matched_item_code = str(best_row[code_col])
        matched_description = str(best_row[desc_col]) if desc_col else ""

        return jsonify(build_response_payload(
            raw_text=raw_text,
            lines=lines,
            detected_code=detected_code,
            detected_batch=detected_batch,
            detected_expiry=detected_expiry,
            detected_gtin=detected_gtin,
            matched_item_code=matched_item_code,
            matched_description=matched_description,
            match_score=best_score,
            match_type=f"fuzzy_{best_match_basis}"
        ))

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
