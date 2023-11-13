import cv2
import numpy as np
from keras.models import load_model
from wide_resnet import WideResNet
from utils.array import scale
from utils.image import crop_bounding_box, draw_bounding_box_with_label

# Constants for model files
FACE_MODEL_FILE = "models\\haarcascade_frontalface_alt.xml"
AG_MODEL_FILE = "models\\weights.18-4.06.hdf5"
EM_MODEL_FILE = 'models\\emotion_model.hdf5'
FACE_SIZE = 64
DEPTH = 16
WIDTH = 8

# Load models
def load_models():
    face_cascade = cv2.CascadeClassifier(FACE_MODEL_FILE)
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    emotion_classifier = load_model(EM_MODEL_FILE)
    emotion_target_size = emotion_classifier.input_shape[1:3]
    # Create an instance of WideResNet using standard instantiation:
    age_gender_model = WideResNet(image_size=FACE_SIZE, depth=DEPTH, k=WIDTH).create_model()
    age_gender_model.load_weights(AG_MODEL_FILE)
    return face_cascade, emotion_labels, emotion_classifier, age_gender_model

# Process frame
def process_frame(frame, face_cascade, emotion_labels, emotion_classifier, age_gender_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for face in faces:
        face_img, cropped = crop_bounding_box(frame, face, margin=0.4, size=(FACE_SIZE, FACE_SIZE))
        (x, y, w, h) = cropped
        age, gender = get_age_gender(face_img, age_gender_model)
        emotion = get_emotion(face_img, emotion_classifier, emotion_labels)
        label = f"{age}, {gender}, {emotion}"
        draw_bounding_box_with_label(frame, x, y, w, h, label)

def get_age_gender(face_image, age_gender_model):
    # Determine the age and gender of the face in the picture
    face_imgs = np.empty((1, FACE_SIZE, FACE_SIZE, 3))
    face_imgs[0, :, :, :] = face_image
    result = age_gender_model.predict(face_imgs)
    est_gender = "F" if result[0][0][0] > 0.5 else "M"
    est_age = int(result[1][0].dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
    return est_age, est_gender

def get_emotion(face_image, emotion_classifier, emotion_labels):
    # Determine the emotion of the face in the picture
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = scale(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    return emotion_labels[emotion_label_arg]

# Main function
def main():
    face_cascade, emotion_labels, emotion_classifier, age_gender_model = load_models()

    # Select video or webcam feed
    USE_WEBCAM = False
    WEBCAM_ID = 0
    VIDEO_FILE = "demo.mp4"

    if USE_WEBCAM:
        capture = cv2.VideoCapture(WEBCAM_ID)
    else:
        capture = cv2.VideoCapture(VIDEO_FILE)

    while capture.isOpened():
        success, video_frame = capture.read()
        if not success:
            print("Error reading frame from the camera")
            break
        process_frame(video_frame, face_cascade, emotion_labels, emotion_classifier, age_gender_model)
        cv2.imshow('Video', video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
