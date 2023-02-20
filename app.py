################# Loading libraries and frameworks #################
from flask import Flask, render_template, Response
import cv2
import os
import json
import imutils
import argparse
from persondetection import DetectorAPI
import matplotlib.pyplot as plt
from fpdf import FPDF
#####################################################################

app = Flask(__name__)

################################## Haar Detector path ##################################
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

HUMAN_DETECTION_PATH = "{base_path}/cascades/haarcascade_fullbody.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
#########################################################################################

############### Haar Classifier creation ###############
face_class = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
########################################################

n_faces = 0

#################### Get video ####################
## Video path. By default, we get the webcam ## 
video_path=0
###################################################
camera = cv2.VideoCapture(video_path)
###################################################

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function which allows us to detect the face #
def detect_face():  
    global max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, max_acc3, max_avg_acc3
    max_count3 = 0
    framex3 = []
    county3 = []
    max3 = []
    avg_acc3_list = []
    max_avg_acc3_list = []
    max_acc3 = 0
    max_avg_acc3 = 0
    
    
    odapi = DetectorAPI()
    threshold = 0.7

    x3 = 0

    cache = ()
    countFrame = 0
    
    count_frame = 0
    count_peaple = 0

    while True:
        check, frame = camera.read()
        img = cv2.resize(frame, (800, 600))
        countFrame = countFrame + 1 

        if countFrame == 1:
            cache = odapi.processFrame(img)

        if(countFrame % 20 != 0):
            boxes, scores, classes, num = cache
        else:
            cache = odapi.processFrame(img)


        if countFrame == 500:
            countFrame = 0

        person = 0
        acc = 0
        for i in range(len(boxes)):

            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                acc += scores[i]
        
                if (scores[i] > max_acc3):
                    max_acc3 = scores[i]

        if (person > max_count3):
            max_count3 = person


        img = cv2.putText(img, str(person), (55,55), 1, 3, (0,255,0))


        key = cv2.waitKey(1)
        

        county3.append(person)
        x3 += 1
        framex3.append(x3)
        if(person>=1):
            avg_acc3_list.append(acc / person)
            if ((acc / person) > max_avg_acc3):
                max_avg_acc3 = (acc / person)
            else:
                avg_acc3_list.append(acc)


        ret, buffer = cv2.imencode('.jpg', img)
        show_frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n') 


######################## Routing to the face detection function ########################
@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################################################

################# Main page #################
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')
#############################################
if __name__ == '__main__':
    app.run(debug=False)
