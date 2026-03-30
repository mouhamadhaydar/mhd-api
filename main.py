from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import easyocr
import io
import os
import torch
import numpy as np

app = Flask(__name__)

# Limit CPU usage (important for GoDaddy / Render)
torch.set_num_threads(1)

MODEL_DIR = os.environ.get("EASYOCR_MODULE_PATH", "/opt/render/project/src/.EasyOCR")

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=True
)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    return np.array(img)

# -----------------------------
# OCR API (TEXT ONLY)
# -----------------------------
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
                "error": "Empty file"
            }), 400

        img = preprocess_image(file_bytes)

        # 🔥 OCR ONLY (no extraction)
        results = reader.readtext(img, detail=0, paragraph=True)

        full_text = " ".join(results).strip()

        return jsonify({
            "success": True,
            "text": full_text,
            "lines": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/")
def home():
    return "OCR TEXT API is running 🚀"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
