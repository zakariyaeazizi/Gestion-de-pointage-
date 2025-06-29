import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Caméra non accessible")
else:
    print("✅ Caméra détectée")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Camera", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
