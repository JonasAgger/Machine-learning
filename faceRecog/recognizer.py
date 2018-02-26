import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

face = cv2.imread("face.jpg", 2)

# cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

collector = rec_lbph.predict(face)

print(collector)

# plt.imshow(face)
# plt.show()

while True:
    cv2.imshow('test', face)
    if ord('q') == cv2.waitKey(20):
        exit(0)
