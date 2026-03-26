import easyocr
import sys
import json
from PIL import Image

def read_image_text(image_path):
    try:
        reader = easyocr.Reader(['en'])  # add 'ar' if Arabic needed
        result = reader.readtext(image_path, detail=0)

        return {
            "success": True,
            "text": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}))
        sys.exit(0)

    image_path = sys.argv[1]
    output = read_image_text(image_path)
    print(json.dumps(output))
