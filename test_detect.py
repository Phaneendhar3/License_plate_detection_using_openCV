import cv2
import detect_plate
import easyocr

img = cv2.imread('car.jpg')
bbox, plate_img, prepared, pts = detect_plate.detect_plate(img)
print('bbox', bbox)
reader = easyocr.Reader(['en'], gpu=False)
if bbox is not None:
    if prepared is not None:
        plate_rgb = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB)
    else:
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(plate_rgb, detail=1)
    print('ocr_results:', results)
    texts = [r[1] for r in results]
    confs = [r[2] for r in results]
    print('texts:', texts)
    print('confs:', confs)
else:
    print('no plate detected')
