from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from playsound import playsound
from time import sleep
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)
text_data  = open('TTS words.txt').readlines()    

for i in range(len(text_data)):
    num,val = text_data[i].split('.')
    text_data[i] = (num,val.strip())
text_data = dict(text_data)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
print('New Capture: ')
while True:
        # Grab the webcamera's image.
    ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    frame = image

        # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
    image = (image / 127.5) - 1

        # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    confidence_score = prediction[0][index]
    gesture_id = class_name.split('_')[1][:-1]
    gesture_name = text_data.get(gesture_id,'Unknown Gesture')
    conf = str(np.round(confidence_score * 100))[:-2]

    if gesture_name != 'Unknown Gesture' and int(conf) > 85:
        frame = cv2.putText(frame, gesture_name, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Webcam Image", frame)
    # Show the image in a window
    if gesture_name !='Unknown Gesture' and int(conf)>90:
        playsound(f'audio/{gesture_id}.mp3')

    # Print prediction and confidence score
    print("Class:", class_name,"Gesture:", gesture_name)
    print("Confidence Score:",conf, "%")
    # Mos AUDIO formatni play qilish

    #===============================
    keyboard_input = cv2.waitKey(1)
        # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

# camera.release()
cv2.destroyAllWindows()
 