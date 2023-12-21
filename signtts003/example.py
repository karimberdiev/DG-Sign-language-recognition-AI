import threading
from keras.models import load_model
import cv2
import numpy as np
from playsound import playsound
import datetime

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)
text_data = open('TTS words.txt').readlines()

for i in range(len(text_data)):
    num, val = text_data[i].split('.')
    text_data[i] = (num, val.strip())

text_data = dict(text_data)
class_names = open("labels.txt", "r").readlines()

print(text_data)
print(class_names)

camera = cv2.VideoCapture(0)

predicted = []
show_text_time = datetime.timedelta(seconds=3)
playing = False


def thread_play(path):
    global playing

    def main():
        global playing
        playing = True
        playsound(path)
        playing = False

    threading.Thread(target=main, daemon=True).start()


last_text = ""
last_text_start_time = datetime.datetime.now()

while True:
    ret, image = camera.read()

    image_showing = cv2.resize(image, (1366, 786))
    image_predict = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_predict = np.asarray(image_predict, dtype=np.float32).reshape(1, 224, 224, 3)
    image_predict = (image_predict / 127.5) - 1

    prediction = list(model.predict(image_predict)[0])

    prediction_max = max(prediction)
    prediction_max_index = prediction.index(prediction_max)

    while True:
        if prediction_max_index in predicted:
            prediction.remove(prediction_max)
            prediction.insert(prediction_max_index, 0.0)

            prediction_max = max(prediction)
            prediction_max_index = prediction.index(prediction_max)
        else:
            break

    prediction_name = text_data[str(prediction_max_index + 1)]

    print(prediction_name, prediction_max, prediction_max_index, prediction, predicted)

    if prediction_max >= 0.08 and prediction_max_index not in predicted:
        if not playing:
            last_text_start_time = datetime.datetime.now()
            last_text = prediction_name
            thread_play(f"audio/{prediction_max_index}.mp3")
            predicted.append(prediction_max_index)

    if (datetime.datetime.now() - last_text_start_time) < show_text_time:
        frame = cv2.putText(image_showing, last_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow("Webcam Image", image_showing)

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break