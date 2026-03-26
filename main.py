from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Render app is running"

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/ocr", methods=["POST"])
def ocr():
    return jsonify({"message": "OCR endpoint ready"})
