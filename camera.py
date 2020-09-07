import cv2

face_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGRA2RGBA)
        faces = face_classifier.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(fr,
                          (x, y),
                          (x + w, y + h),
                          (255, 0, 0),
                          2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
