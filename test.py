import cv2
cap = cv2.VideoCapture(0)

while True:
    istrue, frame =  cap.read()
    cv2.imshow("inco",frame)

    cv2.waitKey(1)
    