import cv2


def normalize_intensity(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)


def cut_face_from_image(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        image = image[y: y + h, x + w_rm: x + w - w_rm]
    return image


def resize_image(image, size=(50, 50)):
    if image.shape < size:
        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    else:
        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image_resized


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

    faces_coord = classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                              minSize=min_size, flags=flags)

    if len(faces_coord):
        newimg = cut_face_from_image(img, faces_coord)

        newimg = normalize_intensity(newimg)

        newimg = resize_image(newimg, (125, 125))

        cv2.imshow("hello", newimg)
    else:
        cv2.imshow("hello", img)

    if ord('q') == cv2.waitKey(10):
        exit(0)
