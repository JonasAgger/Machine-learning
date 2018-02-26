from urllib import request
import cv2
import numpy as np
import os
import sys
from functions import *
"""
if len(sys.argv) < 2:
    print("Needs to give adress to webcam")
    exit(1)

url = "http://100.82.132.229:8080/shot.jpg"
"""
image = cv2.VideoCapture(0)

if not image.isOpened():
    print("no")

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

scale_factor = 1.2
min_neighbours = 5
min_size = (30, 30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE

capture = False
count = 0
personname = ""

while True:
    # imgResp = request.urlopen(url)
    # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    ret, img = image.read() # cv2.imdecode(imgNp, -1)


    faces_coord = classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                              minSize=min_size, flags=flags)


    if len(faces_coord):
        newimg = cut_face_from_image(img, faces_coord)

        newimg = normalize_intensity(newimg)

        newimg = resize_image(newimg, (125, 125))



    cv2.imshow('test', img)

    if capture == True and len(faces_coord) > 0:
        cv2.imwrite(os.path.join(("People/"+personname),(personname+str(count)+".jpg")), img=newimg)
        count += 1
    if count >= 25:
        print("Done")
        capture = False
        count = 0

    switch = cv2.waitKey(50)
    if ord('q') == switch:
        exit(0)
    elif ord('s') == switch:
        cv2.destroyAllWindows()
        personname = input("Enter name of person")
        if not os.path.exists('People/' + personname):
            os.makedirs('People/' + personname)
        capture = True
