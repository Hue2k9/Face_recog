import cv2
import os

from pathlib import Path

# Initialize the classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start the video camera
vc = cv2.VideoCapture(0)

# Get the userId and userName
print("Enter the id and name of the person: ")
userId = input()
userName = input()

# Initially Count is = 1
count = 1

# Function to save the image                             :vv di chuyển con chuột vào đây :vv              
def saveImage(image, userName, userId, imgId):
    # Create a folder with the name as userName
    Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    # Save the images inside the previously created folder
    cv2.imwrite("./dataset/{}/{}_{}.jpg".format(userName, userId, imgId), image)
    print("[INFO] Image {} has been saved in folder : {}".format(
        imgId, userName))


print("[INFO] Video Capture is now starting please stay still...")

 # thẳng mặt lên cho đúng mặt vào khung chứ mới lấy được data ok từ đã tạo thư mục đã ok
vc = cv2.VideoCapture(0)
while True:
    # Capture the frame/image
    res,img = vc.read()
    if img is None:
        break
    # Copy the original Image
    originalImg = img.copy()

# lúc hiện lên rồi thì ấn nút "s " giữ nguyên để lấy data là xong ok ok để test lại xóa folder đấy đi làm lại xem ấn giữ s mà đừng ấn 1 lần ấn giữ
# Giữ nó lưu nhiều à? 5 cái tự thoát mà ok ok :vvvv giữ mà 
    # Get the gray version of our image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the coordinates of the location of the face in the picture
    faces = faceCascade.detectMultiScale(gray_img,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(50, 50))

    # Draw a rectangle at the location of the coordinates
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        coords = [x, y, w, h]

    # Show the image
    cv2.imshow("Identified Face", img)

    # Wait for user keypress
    key = cv2.waitKey(1) & 0xFF

    # Check if the pressed key is 'k' or 'q'
    if key == ord('s'):
        # If count is less than 5 then save the image
        if count <= 5:
            roi_img = originalImg[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]
            saveImage(roi_img, userName, userId, count)
            count += 1
        else:
            break
    # If q is pressed break out of the loop
    elif key == ord('q'):
        break

print("[INFO] Dataset has been created for {}".format(userName))

# Stop the video camera
vc.release()
# Close all Windows
cv2.destroyAllWindows()


# sao đấy 
# À ko tui đang đinh để cái này thành ./