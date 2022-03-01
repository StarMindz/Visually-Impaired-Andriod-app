from flask import Flask, render_template, request, Response, redirect
# import cv2
# from cgitb import text
# from re import T
# from PIL import Image
import numpy as np
# import os
import pytesseract
import urllib.request
from pathlib import Path
# import sys
import time
import pyttsx3

import io
import os
import cv2
# from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types




#connect json file containing the credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="<google-credential-json-file>"   #Add your google creedential json file here
#instantiate a client
client=vision.ImageAnnotatorClient()


ROOT = Path('')
pytesseract.pytesseract.tesseract_cmd = r'tesseract-ocr/tesseract.exe'
#engine = pyttsx3.init()
cfg = 'yolov3-tiny.cfg'
weights = 'yolov3-tiny.weights'
net = cv2.dnn.readNet(weights, cfg)
classes = open("coco.names").read().strip().split("\n")
layer_names = net.getLayerNames()
print(net.getUnconnectedOutLayers())
output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# def text_to_read(text):
#     text = pytesseract.image_to_string(text, lang='eng')
#     print('Texto: ', text)
#     if not text == "":
#         readText(text)
#         return text
#     else:
#         return None



def readText(text="Reading mode"):   #Function for reading out text using pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def offline_detect_text(img):
    text = pytesseract.image_to_string(img, lang='eng')
    print('Texto: ', text)
    if not text == "":
        readText(text)
        return text
    else:
        return None



def offline_object_recognition(img):        #Function for carrying out object detection using YOLO
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (128, 128), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.01:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    label = ""

    # Draw the predicted bounding box
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1.0, color, 2)
            print(label)
    if not label == "":
        label = "There is a {} in front of you".format(label)
        readText(label)
    else:
        pass




    # img.release()p



def online_detect_text(path):
    """Detects text in the file."""
    #image_rotation(path)
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    string = ''

    for text in texts:
        string+=' ' + text.description
    # print("I'm online: "+string)
    if not string == "":
        readText(string)
    else:
        pass


def online_object_recognition(path):   #For object detection
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.object_localization(
        image=image).localized_object_annotations

    #print('Number of objects found: {}'.format(len(response)))
    #print ("response: {}".format(response))
    main_obj=""
    for object_ in response:
        #print(object_.name)
        main_obj=object_.name
        break
    if not main_obj == "":
        main_obj = "There is a {} in front of you".format(main_obj)
        readText(main_obj)
        #print("I am good")
    else:
        pass



def read():     #The function called by Flask framework that allow for text to be read
    #url = "http://192.168.137.65/capture"
    url=0
    if __name__ == '__main__':

        timeout = time.time()+10

        cap = cv2.VideoCapture(url,cv2.CAP_DSHOW)


        while True:

            ret, img = cap.read()
            # img_resp = urllib.request.urlopen(url)
            # imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            # img = cv2.imdecode(imgnp, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            file = 'text1.png'
            cv2.imwrite(file, img)

            try:
                online_detect_text(file)
            except:
                offline_detect_text(img)
            if time.time()>timeout:
                break
        cv2.destroyAllWindows()



def detect():       #The function called by flask framework that does the object detection
    url=0   # use this if using Pc local camera
    #url = "http://192.168.137.65/capture"  #Use this if using ESP32 cam
    if __name__ == '__main__':
        timeout = time.time()+10

        cap = cv2.VideoCapture(url,cv2.CAP_DSHOW)

        last_read =""
        while True:
            ret, img = cap.read()      #This line should be used when using PC local camera
            # img_resp = urllib.request.urlopen(url)     #The next lines should be used when using ESP cam
            # imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            # img = cv2.imdecode(imgnp, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            file = 'text1.png'
            cv2.imwrite(file, img)
            try:
                online_object_recognition(file)
            except:
                #print("I didnt escape")
                offline_object_recognition(img)
            if time.time() > timeout:
                break
        cv2.destroyAllWindows()





app = Flask(__name__)



@app.route("/reader", methods=["POST", "GET"])
def reader():
    print("Read")
    mimetype = "multipart/x-mixed-replace; boundary=frame"
    read()
    return render_template("index.html")


@app.route("/detector", methods=["POST", "GET"])
def detector():
    print("Detect")
    mimetype = "multipart/x-mixed-replace; boundary=frame"
    detect()
    return render_template("index.html")





@app.route("/", methods=["POST", "GET"])
def hello():
    if request.method == "POST":
        if request.form.get("button1") == "Read":
            return redirect('/reader')
        elif request.form.get("button2") == "Detect Objects":
            return redirect('/detector')
        elif request.form["button3"] == "Navigate":
            #print("navigate")
            pass
        else:
            pass

    elif request.method == "GET":
        return render_template("index.html")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


