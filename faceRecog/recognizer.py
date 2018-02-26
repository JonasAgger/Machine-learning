import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir('people/')]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            img = cv2.imread("people/" + person + '/' + image, 2)
            #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(i)
    return (images, np.array(labels), labels_dic)


images, labels, labels_dic = collect_dataset()

rec_lbph = cv2.face.LBPHFaceRecognizer_create()

rec_lbph.train(images, labels)

# collector = cv2.face.StandardCollector_create()

# face = images[25]



classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

scale_factor = 1.2
min_neighbours = 5
min_size = (30, 30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE

image = cv2.VideoCapture(0)

if not image.isOpened():
    print("no")

while True:
    ret, img = image.read()
    face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_coord = classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                              minSize=min_size, flags=flags)

    if len(faces_coord):
        print(len(faces_coord))
        newimg = cut_face_from_image(face, faces_coord)

        newimg = normalize_intensity(newimg)

        newimg = resize_image(newimg, (125, 125))

        collector = rec_lbph.predict(face)

        if collector[1] < 140:
            cv2.putText(img, labels_dic[collector[0]].capitalize(), (faces_coord[0][0],faces_coord[0][1]-10), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            print("Name ", labels_dic[collector[0]], "Confidence ", collector[1])

        for (x, y, w, h) in faces_coord:
            cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 0), 8)


    cv2.imshow('test', img)
    if ord('q') == cv2.waitKey(50):
        exit(0)


"""
# cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

collector = rec_lbph.predict(face)


# plt.imshow(face)
# plt.show()

while True:
    cv2.imshow('test', face)
    if ord('q') == cv2.waitKey(20):
        exit(0)
"""
