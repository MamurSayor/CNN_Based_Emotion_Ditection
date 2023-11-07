import os
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the emotion recognition model
from keras.models import model_from_json

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the Haar Cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Define emotion text descriptions
emotion_text = {
    'angry': 'Feeling Angry',
    'disgust': 'Feeling Disgusted',
    'fear': 'Feeling Fearful',
    'happy': 'Feeling Happy',
    'neutral': 'Feeling Neutral',
    'sad': 'Feeling Sad',
    'surprise': 'Feeling Surprised'
}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def add_circle_to_face(frame, p, q, r, s):
    center = (p + r // 2, q + s // 2)
    radius = max(r, s) // 2
    cv2.circle(frame, center, radius, (0, 255, 0), 3)

def gen_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        try:
            for (p, q, r, s) in faces:
                add_circle_to_face(frame, p, q, r, s)
                image = gray[q:q + s, p:p + r]
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                emotion_description = emotion_text.get(prediction_label, 'Unknown Emotion')
                cv2.putText(frame, emotion_description, (p - 10, q - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except cv2.error:
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
