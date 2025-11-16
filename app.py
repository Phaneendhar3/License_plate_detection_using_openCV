from flask import Flask, render_template, request, jsonify
import io
import base64
from PIL import Image
import numpy as np
import easyocr
import cv2
import detect_plate

app = Flask(__name__)

# Initialize EasyOCR reader (slow first call)
reader = None


def ensure_reader():
    global reader
    if reader is None:
        # use English; EasyOCR is robust for alphanumeric plates
        reader = easyocr.Reader(['en'], gpu=False)
    return reader


def image_to_dataurl(img_bgr):
    # img_bgr: numpy BGR image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil.save(buff, format='JPEG')
    data = base64.b64encode(buff.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{data}'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    # Expecting form with file 'image'
    if 'image' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['image']
    in_memory = file.read()
    npimg = np.frombuffer(in_memory, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400

    # Detect plate bounding box using contour-based detector (no YOLO)
    bbox, plate_img, prepared_for_ocr, plate_pts = detect_plate.detect_plate(img)

    detected_text = ''
    avg_conf = None
    out_img = img.copy()

    if bbox is not None:
        x, y, w, h = bbox
        # Draw only bounding box (no text overlay)
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 158, 73), 4)

        # OCR using EasyOCR on the prepared plate area (prepared_for_ocr is a grayscale image)
        try:
            ensure_reader()
            # If detect_plate returned a preprocessed image, convert to RGB for EasyOCR
            if prepared_for_ocr is not None:
                # prepared_for_ocr is grayscale; convert to 3-channel RGB
                plate_rgb = cv2.cvtColor(prepared_for_ocr, cv2.COLOR_GRAY2RGB)
            else:
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

            results = reader.readtext(plate_rgb, detail=1)
            texts = [r[1] for r in results if r[1].strip()]
            confs = [r[2] for r in results if isinstance(r[2], (int, float))]
            detected_text = ' '.join(texts)
            if confs:
                avg_conf = float(sum(confs) / len(confs))
            # cleanup keep alnum and space
            detected_text = ''.join(ch for ch in detected_text if ch.isalnum() or ch == ' ')
        except Exception as e:
            detected_text = f'OCR error: {e}'
    else:
        detected_text = ''

    # Return image as data URL and text separately; do not save to disk
    data_url = image_to_dataurl(out_img)
    resp = {'image': data_url, 'text': detected_text}
    if avg_conf is not None:
        resp['confidence'] = round(avg_conf, 3)
    return jsonify(resp)


if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=False)
