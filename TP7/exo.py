import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Impossible de lire la webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
   
    edges = cv2.Canny(blur, 50, 100)

    cv2.imshow("Webcam", frame)

    cv2.imshow("Canny", edges)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()