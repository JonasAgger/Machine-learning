import sys
import cv2
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Please provide image name")
    exit(1)

argvStr = sys.argv[1]
	
pic = cv2.imread(argvStr, 1)

if not pic.any():
    print("Not a picture")
    exit(1)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

scale_factor = 1.2
min_neighbours = 5
min_size = (30, 30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE

faces_coord = classifier.detectMultiScale(pic, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                          minSize=min_size, flags=flags)

for (x,y,w,h) in faces_coord:
    cv2.rectangle(pic, (x, y),(x + w, y + h), (150, 150, 0), 8)

#face2 = open("face2.jpg", 'wb')

#face2.write(np.packbits(pic))

if len(pic.shape) == 3:
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

plt.imshow(pic)
plt.show()

pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
newpic = ""
newpic += "facedetect_"
newpic += argvStr

print(newpic)
cv2.imwrite(newpic ,pic)