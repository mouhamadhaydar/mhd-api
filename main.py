from flask import Flask, request, jsonify
import easyocr
import os

app = Flask(__name__)   # ✅ define first

reader = easyocr.Reader(['en'])

@app.route("/")
def home():
    return "OCR API is running 🚀"

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    filepath = "temp.jpg"
    file.save(filepath)

    result = reader.readtext(filepath, detail=0)

    return jsonify({
        "success": True,
        "text": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
