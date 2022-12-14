import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

# Names corresponding to each id
# Them cái này ấy hả 
names = ['Nghia','Hue','Tung']
# for users in os.listdir("dataset/"):
#     names.append(users)

# img = cv2.imread("D:/Python\hue\face_recog\dataset\nghia\1_1.jpg")
# cap = cv2.VideoCapture('D:/Python/hue/face_recog/dataset/nghia/1_1.jpg')
# path = 'C:/Users/laitu/Downloads/face_recog/test_folder/z3946485336444_2ac9a489c5acaf30283b125ec9d161ff.jpg'
# frame = cv2.imread(path)
# gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(0)

# comment dòng này lại rồi bỏ comment 3 dòng trên 
while True:
    # images
    # frame = cv2.imread(path)
    # webcam
    res,frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
    )

    # Try to predict the face and get the id
    # Then check if id == 1 or id == 2
    # Accordingly add the names
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, _ = recognizer.predict(gray_image[y : y + h, x : x + w])
        print(id)
        if id:
            cv2.putText(
                frame,
                names[id-1],
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Unknown",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Recognize", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# video_capture.release()
cv2.destroyAllWindows()

# thêm data thì chạy file generate ( nhập Id và tên)
# sau đó chạy file thì chạy file training.py
# cuối cùng thêm tên vào names [] trong file face_recog.py đây này oke chưa
# chạy để check là file training file face_recog ok okoke nhé
# Cái path link kiểu ./ ok ko?
# cái đấy dùng cho b thử bằng ảnh có ảnh không đưa vào đây thử tét xem 
#Tình hình là chưa có