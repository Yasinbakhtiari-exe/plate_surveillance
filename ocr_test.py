from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_rec_score_thresh=0.3,
)

result = ocr.predict("test.png")

for res in result:
    texts = res["rec_texts"]
    scores = res["rec_scores"]

    print("TEXT:", texts)
    print("SCORES:", scores)

    full_text = "".join(texts)

    print("FULL TEXT:", full_text)