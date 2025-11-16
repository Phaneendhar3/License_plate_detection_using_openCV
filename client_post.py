import requests
import base64
import sys
import webbrowser

URL = 'http://127.0.0.1:5000/detect'
IMG_CANDIDATES = ['sample_car.jpg', 'car.jpg', 'sample.jpg']

def find_image():
    import os
    for p in IMG_CANDIDATES:
        if os.path.exists(p):
            return p
    # fallback: list files and pick first jpg/png
    for f in os.listdir('.'):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return f
    return None


def main():
    img_path = find_image()
    if not img_path:
        print('No image found in workspace to test. Please upload one to the folder and retry.')
        sys.exit(2)

    print('Posting', img_path, 'to', URL)
    with open(img_path, 'rb') as fh:
        files = {'image': fh}
        try:
            r = requests.post(URL, files=files, timeout=15)
        except Exception as e:
            print('Request failed:', e)
            sys.exit(3)

    if r.status_code != 200:
        print('Server returned', r.status_code, r.text)
        sys.exit(4)

    data = r.json()
    imgdata = data.get('image')
    text = data.get('text')
    conf = data.get('confidence')

    if imgdata and imgdata.startswith('data:image'):
        # open the returned image in the default browser without saving
        try:
            webbrowser.open(imgdata)
            print('Opened result image in browser (no file saved).')
        except Exception:
            print('Could not open image in browser; image data is available in response.')
    else:
        print('No image returned')

    print('Detected text:', text)
    if conf is not None:
        print('Confidence:', conf)


if __name__ == '__main__':
    main()
