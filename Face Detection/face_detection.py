import cv2.cv2
import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontal_face_default.xhtml")
camera_vid = cv2.VideoCapture(0)
# camera_vid.set(3, 640)
# camera_vid.set(4, 480)
# camera_vid.set(10, 100)
while True:
    success, img = camera_vid.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

