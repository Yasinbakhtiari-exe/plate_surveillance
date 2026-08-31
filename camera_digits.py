import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

print("Camera started")
print("Q = exit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()