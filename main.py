from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import easyocr
import io
import os
import json
import requests
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

# OpenAI config
OPENAI_API_KEY = os.environ.get("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", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

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
    text = re.sub(r'[^a-z0-9\s./()-]', ' ', text)
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
        df[gtin_col] = df[gtin_col].fillna('').astype(str).str.replace(r'[^0-9]', '', regex=True)

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

    numeric_candidates = re.findall(r'\b\d{5,10}\b', text)
    for num in numeric_candidates:
        if num == gtin:
            continue
        if expiry and num in expiry:
            continue
        if num.startswith('20'):
            continue
        if num in ('300731', '01', '10', '17'):
            continue
        return num

    patterns = [
        r'\b([A-Z]{2,10}\.\d{2,10})\b',
        r'\b([A-Z]{1,5}-\d{2,10})\b',
        r'\b([A-Z]{2,10}\d{2,10})\b'
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = m.group(1)
            if value != gtin:
                return value

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

    skip_exact_tokens = {'C-13'}

    for token in tokens:
        if token in skip_words or token in skip_exact_tokens:
            continue
        if token == detected_code or token == detected_gtin or token == detected_expiry:
            continue

        if re.fullmatch(r'\d{8,14}', token):
            continue
        if re.fullmatch(r'20\d{2}-\d{2}-\d{2}', token):
            continue
        if re.fullmatch(r'[A-Z]{2,10}\.\d{2,10}', token):
            continue
        if re.fullmatch(r'[A-Z]{1,5}-\d{2,10}', token):
            continue
        if re.fullmatch(r'[A-Z]{2,10}\d{2,10}', token):
            continue
        if re.fullmatch(r'\d+(\.\d+)?(MM|CM|M)?', token):
            continue
        if re.fullmatch(r'\d+/\d+', token):
            continue
        if re.fullmatch(r'[A-Z]-\d{1,3}', token):
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


# ----------------------------
# OPENAI FALLBACK
# ----------------------------
def build_candidate_list(df, code_col, desc_col, gtin_col, detected_code, detected_gtin, raw_text, top_n=20):
    text_clean = clean_text(raw_text)
    candidates = []

    for _, row in df.iterrows():
        code_value = str(row[code_col]) if pd.notna(row[code_col]) else ""
        desc_value = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else ""
        gtin_value = str(row[gtin_col]) if gtin_col and pd.notna(row[gtin_col]) else ""

        combined = f"{code_value} {desc_value} {gtin_value}"
        combined_clean = clean_text(combined)

        score_code = fuzz.partial_ratio(clean_text(detected_code), clean_text(code_value)) if detected_code and code_value else 0
        score_gtin = 100 if detected_gtin and gtin_value and detected_gtin == re.sub(r'[^0-9]', '', gtin_value) else 0
        score_text = fuzz.partial_ratio(text_clean, combined_clean) if combined_clean else 0

        score = max(score_code, score_gtin, score_text)

        candidates.append({
            "item_code": code_value,
            "description": desc_value,
            "gtin": re.sub(r'[^0-9]', '', gtin_value),
            "score": round(float(score), 2)
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def extract_text_output_from_responses_api(data):
    """
    Best-effort parser for Responses API output.
    """
    # Common newer shape
    if isinstance(data, dict):
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()

        output = data.get("output", [])
        collected = []

        for item in output:
            content = item.get("content", [])
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("output_text", "text"):
                        txt = part.get("text", "")
                        if txt:
                            collected.append(txt)

        if collected:
            return "\n".join(collected).strip()

    return ""


def call_openai_product_fallback(raw_text, detected_code, detected_batch, detected_expiry, detected_gtin,
                                 csv_candidates):
    if not OPENAI_API_KEY:
        return {
            "used": False,
            "error": "OPENAI_API_KEY is missing"
        }

    system_prompt = """
You are helping identify a medical or warehouse product from OCR text.

Return STRICT JSON only with this schema:
{
  "matched_item_code": "string",
  "matched_description": "string",
  "gtin": "string",
  "batch": "string",
  "expiry_date": "string",
  "confidence": 0,
  "reason": "string",
  "is_unknown": false
}

Rules:
- Use the OCR text and detected fields carefully.
- Prefer exact evidence from OCR text.
- Prefer any candidate whose GTIN exactly matches.
- If you are not confident, set "is_unknown": true.
- Do not invent data not supported by OCR text or candidates.
- confidence must be 0 to 100.
- Output JSON only, no markdown.
""".strip()

    user_payload = {
        "ocr_text": raw_text,
        "detected_fields": {
            "detected_code": detected_code,
            "batch": detected_batch,
            "expiry_date": detected_expiry,
            "gtin": detected_gtin
        },
        "csv_candidate_matches": csv_candidates
    }

    body = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        "max_output_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=body,
        timeout=60
    )

    resp.raise_for_status()
    data = resp.json()

    text_output = extract_text_output_from_responses_api(data)
    if not text_output:
        return {
            "used": False,
            "error": "OpenAI returned no text output",
            "raw_response": data
        }

    try:
        parsed = json.loads(text_output)
    except Exception:
        return {
            "used": False,
            "error": "OpenAI output was not valid JSON",
            "raw_text": text_output
        }

    return {
        "used": True,
        "data": parsed
    }


def merge_ai_fallback_result(raw_text, lines, detected_code, detected_batch, detected_expiry, detected_gtin,
                             ai_result):
    ai = ai_result.get("data", {}) if isinstance(ai_result, dict) else {}

    ai_item_code = str(ai.get("matched_item_code", "")).strip()
    ai_description = str(ai.get("matched_description", "")).strip()
    ai_gtin = str(ai.get("gtin", "")).strip()
    ai_batch = str(ai.get("batch", "")).strip()
    ai_expiry = str(ai.get("expiry_date", "")).strip()
    ai_confidence = ai.get("confidence", 0)
    ai_reason = str(ai.get("reason", "")).strip()
    ai_unknown = bool(ai.get("is_unknown", False))

    final_gtin = ai_gtin or detected_gtin
    final_batch = ai_batch or detected_batch
    final_expiry = ai_expiry or detected_expiry

    if ai_unknown or (not ai_item_code and not ai_description):
        return {
            "success": True,
            "ocr_text": raw_text,
            "lines": lines,
            "detected_code": detected_code,
            "matched_item_code": "UNKNOWN PRODUCT",
            "matched_description": "UNKNOWN PRODUCT",
            "batch": final_batch,
            "expiry_date": final_expiry,
            "gtin": final_gtin,
            "match_score": 0,
            "match_type": "unknown_product_openai_checked",
            "openai_used": True,
            "openai_confidence": ai_confidence,
            "openai_reason": ai_reason
        }

    return {
        "success": True,
        "ocr_text": raw_text,
        "lines": lines,
        "detected_code": detected_code,
        "matched_item_code": ai_item_code or "UNKNOWN PRODUCT",
        "matched_description": ai_description or "UNKNOWN PRODUCT",
        "batch": final_batch,
        "expiry_date": final_expiry,
        "gtin": final_gtin,
        "match_score": float(ai_confidence) if str(ai_confidence).strip() != "" else 0,
        "match_type": "openai_fallback",
        "openai_used": True,
        "openai_confidence": ai_confidence,
        "openai_reason": ai_reason
    }


# ----------------------------
# ROUTES
# ----------------------------
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
        detected_batch = detect_batch(raw_text)
        detected_expiry = detect_expiry_date(raw_text)
        detected_gtin = detect_gtin(raw_text)

        df, code_col, desc_col, gtin_col = load_items(CSV_FILE_PATH)

        MIN_MATCH_SCORE = 70

        # 1) GTIN exact match
        gtin_row = find_exact_gtin_match(detected_gtin, df, gtin_col)
        if gtin_row is not None:
            return jsonify({
                "success": True,
                "ocr_text": raw_text,
                "lines": lines,
                "detected_code": detected_code,
                "matched_item_code": str(gtin_row[code_col]),
                "matched_description": str(gtin_row[desc_col]) if desc_col else "",
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin,
                "match_score": 100.0,
                "match_type": "gtin_exact",
                "openai_used": False
            })

        # 2) item code exact match
        exact_row = find_exact_code_match(detected_code, df, code_col)
        if exact_row is not None:
            return jsonify({
                "success": True,
                "ocr_text": raw_text,
                "lines": lines,
                "detected_code": detected_code,
                "matched_item_code": str(exact_row[code_col]),
                "matched_description": str(exact_row[desc_col]) if desc_col else "",
                "batch": detected_batch,
                "expiry_date": detected_expiry,
                "gtin": detected_gtin,
                "match_score": 100.0,
                "match_type": "exact_code",
                "openai_used": False
            })

        # 3) fuzzy compare against CSV
        best_row, best_score, best_match_basis = find_best_item_match(raw_text, df, code_col, desc_col)

        # 4) if score too low => try OpenAI fallback
        if best_row is None or float(best_score) < MIN_MATCH_SCORE:
            candidates = build_candidate_list(
                df=df,
                code_col=code_col,
                desc_col=desc_col,
                gtin_col=gtin_col,
                detected_code=detected_code,
                detected_gtin=detected_gtin,
                raw_text=raw_text,
                top_n=20
            )

            try:
                ai_result = call_openai_product_fallback(
                    raw_text=raw_text,
                    detected_code=detected_code,
                    detected_batch=detected_batch,
                    detected_expiry=detected_expiry,
                    detected_gtin=detected_gtin,
                    csv_candidates=candidates
                )

                if ai_result.get("used"):
                    return jsonify(merge_ai_fallback_result(
                        raw_text=raw_text,
                        lines=lines,
                        detected_code=detected_code,
                        detected_batch=detected_batch,
                        detected_expiry=detected_expiry,
                        detected_gtin=detected_gtin,
                        ai_result=ai_result
                    ))

                return jsonify({
                    "success": True,
                    "ocr_text": raw_text,
                    "lines": lines,
                    "detected_code": detected_code,
                    "matched_item_code": "UNKNOWN PRODUCT",
                    "matched_description": "UNKNOWN PRODUCT",
                    "batch": detected_batch,
                    "expiry_date": detected_expiry,
                    "gtin": detected_gtin,
                    "match_score": round(float(best_score), 2) if best_row is not None else 0,
                    "match_type": "unknown_product",
                    "openai_used": False,
                    "openai_error": ai_result.get("error", "OpenAI fallback failed")
                })

            except Exception as ai_ex:
                return jsonify({
                    "success": True,
                    "ocr_text": raw_text,
                    "lines": lines,
                    "detected_code": detected_code,
                    "matched_item_code": "UNKNOWN PRODUCT",
                    "matched_description": "UNKNOWN PRODUCT",
                    "batch": detected_batch,
                    "expiry_date": detected_expiry,
                    "gtin": detected_gtin,
                    "match_score": round(float(best_score), 2) if best_row is not None else 0,
                    "match_type": "unknown_product",
                    "openai_used": False,
                    "openai_error": str(ai_ex)
                })

        # 5) valid fuzzy match
        matched_item_code = str(best_row[code_col])
        matched_description = str(best_row[desc_col]) if desc_col else ""

        return jsonify({
            "success": True,
            "ocr_text": raw_text,
            "lines": lines,
            "detected_code": detected_code,
            "matched_item_code": matched_item_code,
            "matched_description": matched_description,
            "batch": detected_batch,
            "expiry_date": detected_expiry,
            "gtin": detected_gtin,
            "match_score": round(float(best_score), 2),
            "match_type": f"fuzzy_{best_match_basis}",
            "openai_used": False
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# RUN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
