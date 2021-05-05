import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime,timedelta

from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer_e100.json", "r").read())
#load weights
model.load_weights('fer_e100.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name,emo):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        
        nameList=[]
        
        for index,line in enumerate(myDataList):
            entry = line.split(',')
            #print('Entry:', entry)
            #print('Line', line)
            #print('LAST LINE: ',myDataList[-1])
            #print('Emotion: ',emo)
            #nameList.append(entry[0]) #removing repetition in a file
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%D')
                tString=now.strftime('%H:%M:%S')
                #f.writelines(f'\n{name},{dtString},{tString},{emo}')
                #below for per second entry and ingored milisec
                if((myDataList[-1].split(',')[0]==line.split(',')[0]) and (myDataList[-1].split(',')[-1]==line.split(',')[-1])):
                    f.writelines(f'\n{name},{dtString},{tString},{emo}')
                
               #if((myDataList[-1].split(',')[0]==line.split(',')[0]) and (myDataList[-1].split(',')[-1]==line.split(',')[-1])):
                    #f.writelines(f'\n{name},{dtString},{tString},{'emo')

"""Entry: ['ABHINAV SRIVASTAVA', '04/16/21', '13:56:16\n']
Line ABHINAV SRIVASTAVA,04/16/21,13:56:16

Entry: ['ABHINAV SRIVASTAVA', '04/16/21', '13:56:16\n']
Line ABHINAV SRIVASTAVA,04/16/21,13:56:16

Entry: ['ABHINAV SRIVASTAVA', '04/16/21', '13:56:16\n']
Line ABHINAV SRIVASTAVA,04/16/21,13:56:16"""


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    ret,img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
	#
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels=  img_pixels-np.mean(img_pixels)
        img_pixels=  img_pixels/np.std(img_pixels)

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        emo=predicted_emotion
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	
    resized_img = cv2.resize(img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)
	
	
	#
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
	#print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #name1=name.extend(list(name))
            #name1=set(name1)
            #name1=list(name1)
	#print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #cam = cv2.VideoCapture(0)
            #image = cam.read()[1]
            markAttendance(name,emo)
	#
	
	
	
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows