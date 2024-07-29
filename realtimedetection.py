import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        cv2.putText(im, '%s' % (prediction_label), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Output", im)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
