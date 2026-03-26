from flask import Flask, request, jsonify
import easyocr
import os

app = Flask(__name__)

reader = easyocr.Reader(['en'])

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    filepath = os.path.join("temp.jpg")
    file.save(filepath)

    try:
        result = reader.readtext(filepath, detail=0)

        return jsonify({
            "success": True,
            "text": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
