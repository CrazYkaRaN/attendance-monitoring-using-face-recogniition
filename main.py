import face_recognition as fr
import numpy as np
import cv2
import os
from datetime import datetime

images=[]
studentName=[]
path="students_images"
studentList=os.listdir(path) #return list of names of items in path directory
for student in studentList:
    curImg=cv2.imread(f'{path}/{student}')
    images.append(curImg)
    studentName.append(os.path.splitext(student)[0]) #stores only file name not extentions

#function to find encodings of images of students
def findEncodings(images):
    encodings=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        enc=fr.face_encodings(img)[0]
        encodings.append(enc)
    return encodings

#marking attendence function
def markAttendence(name):
    with open('attendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

encodedStudents=findEncodings(images)
print('enconding completed')

#LIVE webcam
cap=cv2.VideoCapture(0) # 0 is id
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,  (0,0)   ,  None, 0.25,0.25)
    #...................pixelsize...........scale....
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesInCurrentFrame=fr.face_locations(imgS)
    encodingsOfCurrFrame=fr.face_encodings(imgS,facesInCurrentFrame)

    for currFaceLoc,currEncoding in zip(facesInCurrentFrame,encodingsOfCurrFrame):
        matches=fr.compare_faces(encodedStudents,currEncoding)
        faceDist=fr.face_distance(encodedStudents,currEncoding)
        print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=studentName[matchIndex]
            y1,x2,y2,x1=currFaceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)