import argparse
import cv2
import numpy as np
import pytesseract
import os




def sort_contours(cnts, reverse=False):
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=reverse))
    return cnts


def order_points(pts):
    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_plate(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    edged = cv2.Canny(gray, 30, 200)

    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

    plate_cnt = None
    plate_bbox = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # look for quadrilateral
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h > 0 else 0
            area = w * h
            # heuristic checks for license plate shape
            if area > 2000 and 2.0 <= aspect <= 6.5:
                plate_cnt = approx
                plate_bbox = (x, y, w, h)
                break

    if plate_bbox is None:
        # fallback: try rectangular contours by aspect using bounding boxes
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0
            area = w * h
            if area > 2000 and 2.0 <= aspect <= 6.5:
                plate_bbox = (x, y, w, h)
                plate_cnt = None
                break

    if plate_bbox is None:
        return None, None, None, None

    x, y, w, h = plate_bbox
    plate_img = image[y:y+h, x:x+w]

    # If we have 4-point contour, produce a top-down warped plate for OCR
    plate_warped = None
    if plate_cnt is not None:
        try:
            pts = plate_cnt.reshape(4, 2).astype('float32')
            plate_warped = four_point_transform(image, pts)
        except Exception:
            plate_warped = None

    # Fallback: try to deskew using minAreaRect on the cropped plate
    if plate_warped is None:
        try:
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts_p, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_p:
                c = max(cnts_p, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype('int')
                # map box to coordinates relative to full image
                box[:, 0] = np.clip(box[:, 0], 0, plate_img.shape[1]-1)
                box[:, 1] = np.clip(box[:, 1], 0, plate_img.shape[0]-1)
                # if box is valid, warp
                try:
                    plate_warped = four_point_transform(plate_img, box.astype('float32'))
                except Exception:
                    plate_warped = plate_img
            else:
                plate_warped = plate_img
        except Exception:
            plate_warped = plate_img

    # Preprocess for OCR: convert to grayscale, enhance
    plate_gray = cv2.cvtColor(plate_warped, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.resize(plate_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return plate_bbox, plate_img, plate_thresh, plate_cnt


def ocr_plate(plate_img):
    # Tesseract config: single line, alnum characters
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(plate_img, config=config)
    # basic cleanup
    text = text.strip()
    # remove non-alnum
    text = ''.join([c for c in text if c.isalnum()])
    return text

def draw_result(image, bbox, text):
    out = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if text:
        # draw filled rectangle for text background
        cv2.rectangle(out, (x, y - 30), (x + w, y), (0, 255, 0), -1)
        cv2.putText(out, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Detect license plate and perform OCR.")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", required=False, help="Path to save annotated output image")
    parser.add_argument("--tesseract", required=False, help="Path to tesseract executable (optional, helps on Windows)")
    args = parser.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    if not os.path.isfile(args.input):
        print(f"Input file does not exist: {args.input}")
        return

    image = cv2.imread(args.input)
    if image is None:
        print("Failed to read input image (cv2.imread returned None)")
        return

    bbox, plate_img, prepared = detect_plate(image)
    recognized = ""
    if bbox is not None and prepared is not None:
        try:
            recognized = ocr_plate(prepared)
        except Exception as e:
            print("Tesseract OCR failed:", e)
            recognized = ""

    if bbox is not None:
        out = draw_result(image, bbox, recognized)
    else:
        out = image.copy()
        cv2.putText(out, "Plate not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if args.output:
        cv2.imwrite(args.output, out)
        print(f"Wrote annotated image to {args.output}")
    else:
        # show image in a window
        cv2.imshow("Result", out)
        print("Press any key on the image window to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Detected text:", recognized)


if __name__ == "__main__":
    main()
