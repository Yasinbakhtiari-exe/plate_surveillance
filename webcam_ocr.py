import cv2
import re
from datetime import datetime
from collections import Counter

from paddleocr import PaddleOCR


# OCR
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,

    text_rec_score_thresh=0.3,
)



def normalize_digits(text):

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    )

    return text.translate(translation)



def extract_digits(text):

    text = normalize_digits(text)

    return re.sub(r"[^0-9]", "", text)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam started.")
print("Put numbers inside the green box.")
print("Press Q to exit.")


history = []

HISTORY_SIZE = 12
MIN_VOTES = 7

last_logged = ""



while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break



    h, w = frame.shape[:2]

    box_w = int(w * 0.90)
    box_h = int(h * 0.35)

    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2

    x2 = x1 + box_w
    y2 = y1 + box_h



    roi = frame[y1:y2, x1:x2]



    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.5,
        beta=0
    )

    processed = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


    result = ocr.predict(processed)


    detected_parts = []

    for res in result:

        texts = res["rec_texts"]
        scores = res["rec_scores"]

        for text, score in zip(texts, scores):

            if score >= 0.3:

                digits = extract_digits(text)

                if digits:

                    detected_parts.append(digits)


    detected_text = "".join(detected_parts)


    if detected_text:

        print(
            f"OCR: {detected_text}"
        )



    if len(detected_text) >= 3:

        history.append(detected_text)

        if len(history) > HISTORY_SIZE:
            history.pop(0)



    stable_text = ""

    if len(history) >= MIN_VOTES:

        counter = Counter(history)

        most_common_text, votes = counter.most_common(1)[0]

        if votes >= MIN_VOTES:

            stable_text = most_common_text



    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )



    if stable_text:

        cv2.putText(
            frame,
            "Detected: " + stable_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )



        if stable_text != last_logged:

            now = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(
                "plate_log.txt",
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"[LOG] {now} | {stable_text}\n"
                )

            print(
                f"[LOG] {now} | {stable_text}"
            )

            last_logged = stable_text

    else:

        cv2.putText(
            frame,
            "Detecting...",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
    # مپلا تغییر
    cv2.imshow(
        "Plate OCR",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()