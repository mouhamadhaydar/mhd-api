from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import easyocr
import io
import os
import re
import torch

app = Flask(__name__)

# Optional: helps small CPU instances
torch.set_num_threads(1)

# EasyOCR model folder
MODEL_DIR = os.environ.get("EASYOCR_MODULE_PATH", "/opt/render/project/src/.EasyOCR")

# Load OCR reader once at startup
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=True
)

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")

    max_w = 1600
    if img.width > max_w:
        ratio = max_w / float(img.width)
        img = img.resize((max_w, int(img.height * ratio)))

    img = ImageOps.autocontrast(img)
    return img

def extract_fields(lines):
    text = " ".join(lines)

    lot_match = re.search(r'\b(?:LOT|Lot|lot)[:\s-]*([A-Za-z0-9\-\/]+)\b', text)
    exp_match = re.search(r'\b(?:EXP|Expiry|EXPIRY|Exp)[:\s-]*([A-Za-z0-9\/\-]+)\b', text)

    item_code = ""
    for line in lines:
        s = line.strip()
        if re.fullmatch(r'[A-Z0-9][A-Z0-9._\-\/]{2,}', s):
            item_code = s
            break

    return {
        "item_code": item_code,
        "lot": lot_match.group(1) if lot_match else "",
        "expiry": exp_match.group(1) if exp_match else ""
    }

@app.route("/")
def home():
    return "OCR API is running 🚀"

@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method == "GET":
        return """
        <h2>OCR Upload</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <button type="submit">Upload</button>
        </form>
        """

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]
    data = file.read()

    if not data:
        return jsonify({
            "success": False,
            "error": "Empty file"
        }), 400

    try:
        img = preprocess_image(data)

        result = reader.readtext(
            img,
            detail=0,
            paragraph=False,
            batch_size=1
        )

        fields = extract_fields(result)

        return jsonify({
            "success": True,
            "text": result,
            "item_code": fields["item_code"],
            "lot": fields["lot"],
            "expiry": fields["expiry"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
