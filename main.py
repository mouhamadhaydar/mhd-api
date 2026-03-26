from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "OCR API is running 🚀"

@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method == "GET":
        return """
        <h2>OCR Upload</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Upload</button>
        </form>
        """

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    filepath = "temp.jpg"
    file.save(filepath)

    return jsonify({
        "success": True,
        "message": "File uploaded successfully"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
