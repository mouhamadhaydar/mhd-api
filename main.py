from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import easyocr
import io
import os
import re
import torch
import numpy as np

# import from your separate file
from suiteql_from_excel import fetch_suiteql

app = Flask(__name__)

torch.set_num_threads(1)

MODEL_DIR = os.environ.get("EASYOCR_MODULE_PATH", "/opt/render/project/src/.EasyOCR")
EXCEL_PATH = os.environ.get("ERP_EXCEL_PATH", "ERP_NLQ_Training_Dataset_Expanded.xlsx")

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=True
)


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = ImageOps.autocontrast(img)
    return np.array(img)


def clean_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_fields(lines):
    raw_text = " ".join(lines)
    raw_text = clean_text(raw_text)

    item_code = ""
    lot = ""
    expiry = ""

    patterns_item = [
        r"\bitem[:\s]+([A-Za-z0-9._\-/]+)\b",
        r"\bcode[:\s]+([A-Za-z0-9._\-/]+)\b",
        r"\bsku[:\s]+([A-Za-z0-9._\-/]+)\b",
        r"\b([A-Z]{2,}[A-Z0-9._\-/]*)\b"
    ]

    for p in patterns_item:
        m = re.search(p, raw_text, re.IGNORECASE)
        if m:
            item_code = m.group(1)
            break

    m_lot = re.search(r"\b(?:lot|batch)[:\s]+([A-Za-z0-9._\-/]+)\b", raw_text, re.IGNORECASE)
    if m_lot:
        lot = m_lot.group(1)

    m_exp = re.search(r"\b(?:exp|expiry|expire|edate)[:\s]+([A-Za-z0-9/\-]+)\b", raw_text, re.IGNORECASE)
    if m_exp:
        expiry = m_exp.group(1)

    return {
        "item_code": item_code,
        "lot": lot,
        "expiry": expiry,
        "raw_text": raw_text
    }


@app.route("/")
def home():
    return "OCR + SuiteQL API is running 🚀"


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({
                "success": False,
                "error": "Uploaded file is empty"
            }), 400

        img = preprocess_image(file_bytes)
        results = reader.readtext(img, detail=0, paragraph=False)
        lines = [clean_text(x) for x in results if clean_text(x)]

        extracted = extract_fields(lines)

        return jsonify({
            "success": True,
            "ocr_lines": lines,
            "extracted": extracted
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/suiteql", methods=["POST"])
def suiteql():
    try:
        data = request.get_json(silent=True) or {}
        user_query = str(data.get("query", "")).strip()

        if not user_query:
            return jsonify({
                "success": False,
                "error": "Query is required"
            }), 400

        result = fetch_suiteql(user_query, EXCEL_PATH)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/ocr-to-suiteql", methods=["POST"])
def ocr_to_suiteql():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        action = request.form.get("action", "show stock for item {item_code}")
        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({
                "success": False,
                "error": "Uploaded file is empty"
            }), 400

        img = preprocess_image(file_bytes)
        results = reader.readtext(img, detail=0, paragraph=False)
        lines = [clean_text(x) for x in results if clean_text(x)]

        extracted = extract_fields(lines)
        item_code = extracted.get("item_code", "").strip()

        if not item_code:
            return jsonify({
                "success": False,
                "error": "Could not detect item code from OCR",
                "ocr_lines": lines,
                "extracted": extracted
            }), 400

        user_query = action.format(item_code=item_code)
        suiteql_result = fetch_suiteql(user_query, EXCEL_PATH)

        return jsonify({
            "success": True,
            "ocr_lines": lines,
            "extracted": extracted,
            "generated_query": user_query,
            "suiteql_result": suiteql_result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
