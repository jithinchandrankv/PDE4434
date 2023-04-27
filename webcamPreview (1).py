import cv2

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()

    display_text = "Uno Card"    
    cv2.putText(frame, display_text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Webcam Preview', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
