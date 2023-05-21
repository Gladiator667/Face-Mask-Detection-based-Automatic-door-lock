from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import os
import cv2
import paho.mqtt.client as paho
import sys


client = paho.Client()
message_received = False
if client.connect("192.168.247.205", 1883, 60) != 0:
    print("Could not connect to MQTT Broker!")
    sys.exit(-1)

print("Connected to the MQTT Broker")



def on_message(client, userdata, message):
    print("Message received")
    global message_received
    message_received = True


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("esp8266/commandToModel")


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)\
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

client.on_connect = on_connect
client.on_message = on_message

client.loop_start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    label = ""
    for (box, pred) in zip(locs, preds):
   
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        if mask > withoutMask:
            label = "Mask"
            if message_received:
                message_received = False
                print("Opening gate")
                client.publish("python/commandToGate", "Mask", 0)
            else:
                print("Please wait for the previous person to complete the process")
        else:
            label = "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        client.loop_stop()
        break
cv2.destroyAllWindows()
vs.stop()
