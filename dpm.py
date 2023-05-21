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

def dpm(frm, face_net, mask_net):
    (h, w) = frm.shape[:2]
    blb = cv2.dnn.blobFromImage(frm, 1.0, (224, 224), (104.0, 177.0, 123.0))

    face_net.setInput(blb)
    dtcts = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, dtcts.shape[2]):
        confidence = dtcts[0, 0, i, 2]
        if confidence > 0.8:
            bx = dtcts[0, 0, i, 3:7] * np.array([width, height, width, height])
            (strtX, strtY, endX, endY) = bx.astype("int")
            (strtX, strtY) = (max(0, strtX), max(0, strtY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))
            face = frm[strtY:endY, strtX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((strtX, strtY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)
    return (locs, preds)

prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

mask_net = load_model("mask_detector.model")

print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()

client.on_connect = on_connect
client.on_message = on_message

client.loop_start()
while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=800)
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    output = ""
    for (bx, pred) in zip(locs, preds):
        (strtX, strtY, endX, endY) = bx
        (mask, without_mask) = pred

        if mask > without_mask:
            output = "Mask"
            if message_received:
                message_received = False
                print("Opening gate")
                client.publish("python/commandToGate", "Mask", 0)
            else:
                print("wait for process completion")
        else:
            output = "No Mask"
        clr = (0, 255, 0) if output == "Mask" else (0, 0, 255)

        output = "{}: {:.2f}%".format(output, max(mask, without_mask) * 100)

        cv2.putText(frame, output, (strtX, strtY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
        cv2.rectangle(frame, (strtX, strtY), (endX, endY), clr, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        client.loop_stop()
        break
cv2.destroyAllWindows()
video_stream.stop()
