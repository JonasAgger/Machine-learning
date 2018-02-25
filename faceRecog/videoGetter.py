from urllib import request
import cv2
import numpy as np
import os


url = "http://100.82.132.229:8080/shot.jpg"

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
    imgResp = request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)


    faces_coord = classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                              minSize=min_size, flags=flags)

    for (x, y, w, h) in faces_coord:
        cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 0), 8)



    cv2.imshow('test', img)

    if capture == True and len(faces_coord) > 0:
        cv2.imwrite(os.path.join(("./"+personname),(personname+str(count)+".jpg")), img=img)
        count += 1
    if count >= 50:
        capture = False
        count = 0

    switch = cv2.waitKey(10)
    if ord('q') == switch:
        exit(0)
    elif ord('s') == switch:
        print("Hello world")
        cv2.destroyAllWindows()
        personname = input("Enter name of person")
        if not os.path.exists(personname):
            os.makedirs(personname)
        capture = True
