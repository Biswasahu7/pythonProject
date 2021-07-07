# ********************************************************************************************
# CAMERA_1 & MODEL_1 CODE DETAILS. (CODE DETECTION FROM WAGON AND MAPPING WITH IR DATA)
# ********************************************************************************************

# All Camera's IP address Details:...
# **************************************

# Cam 1 IP = 192.168.2.241
# Cam 2 IP = 192.168.2.242
# Cam 3 IP = 192.168.2.243

# All Camera's User name and Password:...
# *****************************************

# User Name - admin
# Password - Password@123

# Git addess

# Importing required libraries...
import datetime
import cv2 as cv2
# import touch
import logging
from logging.handlers import TimedRotatingFileHandler
import time
# import easyocr
import numpy as np
import re
import pyodbc
import pandas as pd
from hikvisionapi import Client
from OCR_Model import Easy_OCR
from SQL_Data_Insert import insert_data
from SQL_Data_Insert import Deletedata
# Assigning IP address to variable...
ip = "192.168.2.241"
ip1 = "192.168.2.242"
ip2 = "192.168.2.243"

# Assigning all variable to model_1...
detcode = []
sec = 0.0
framerate = 1.0
out = None
mapping = 0
irmapped = 0
t1 = time.time()
fullcode = ""
last3dig = 0
blankocr = 0
save_image = 0
terminate = 0
OCRresult = 0
couplingcount = 0
digit_9 = 0
digit_10 = 0
allcode = []
Wagon_number = 0
ctime = datetime.datetime.now()
coupling = 0
l2 = datetime.datetime.now()
finalcode = []
temp_list1 = []
insertdata = []
codelist = []

# LOGGER INFO FOR CODE REFERENCE... (Debug checking)
l1 = datetime.datetime.now()
logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler("/home/jsw/WagonNumber_Detection/Model_Logs/logeer_{}.log".format(l1), when="m",interval=60)
logger.addHandler(handler)
l2 = datetime.datetime.now()

# DEFINING EASY_OCR with language English...
# reader = easyocr.Reader(['en'])

print("Start....!")

# Assigning YOLO MODEL into our net variable...
# net = cv2.dnn.readNet('/home/jsw/PycharmProjects/pythonProject/yolov3-config/yolov3_training_final.weights',
#                       '/home/jsw/PycharmProjects/pythonProject/yolov3-config/yolov3_testing.cfg')

# net = cv2.dnn.readNet('/home/jsw/PycharmProjects/pythonProject/yolov3-config/yolov3_training_final.weights',
#                       '/home/jsw/PycharmProjects/pythonProject/yolov3-config/yolov3_testing.cfg')

net = cv2.dnn.readNet('/home/jsw/WagonNumber_Detection/YoloV3_Config/new/yolov3_training_15000.weights',
                      '/home/jsw/WagonNumber_Detection/YoloV3_Config/new/yolov3_testing.cfg')

# Assigning class for the model to detect...
classes = []
with open("/home/jsw/WagonNumber_Detection/YoloV3_Config/new/classes.txt","r") as f: classes = f.read().splitlines()

# Using opencv to capture images from camera...
cap = cv2.VideoCapture()

# cap = cv2.VideoCapture("rtsp://admin:Password@123@192.168.2.241:554/axis-media/media.amp")
# cap = cv2.VideoCapture('rtsp://admin:Password@123@192.168.2.2241/H264?ch=1&subtype=0')

# Camera connection using credentials details...
# cap.open("rtsp://admin:Password@123@{}/Streaming/channels/1/?tcp".format(ip))

cam = Client('http://192.168.2.241', 'admin', 'Password@123')
print("Camera Connected......")

# logger.info("Camera connected.......")

# cap.open('/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/5_video_2021-04-20_Night_02_40_05.330307.avi')
# cap.open('/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/3_video_2021-04-20_Night_00_42_05.305308.avi')
# cap.open('/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/Videos/5_video_2021-04-20_Night_02_40_05.330307.avi')

# CREATE TEXT FILE WHERE WE WILL SAVE THE RESULT...
# logger.info("Creating text file")
# touch.touch("/home/vert/JSW_Project/01_WAGON_NUMBER_TRACKING_Vallari/Images_Videos/Cam_1_INFO/OCR_Result/{}.txt".format(datetime.datetime.now()))
# logger.info("Text File has been Created")

# Running while loop into the camera_1 to perform our logic...
while True:

    # Model is Trying to get live images from camera_1...
    try:
        print("Alive - {}".format(datetime.datetime.now()))

        vid = cam.Streaming.channels[102].picture(method='get', type='opaque_data')
        bytes = b''

        with open('screen1.jpg', 'wb') as f:
            for chunk in vid.iter_content(chunk_size=1024):
                bytes += chunk
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b + 2]
                    bytes = bytes[b + 2:]
                    img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # cv2.imwrite("/media/jsw/Data/newimage/Cam_image_{}.jpg".format(save_image), img)
                    save_image += 1
                    # print("image capture")

                    # GET CURRENT TIME/SEC
                    t2=time.time()
                    if t2-t1>3540:
                        # logger.info("Its 1 hr...will create new video")
                        t1=time.time()
                        out=None

                    # Reading Images from live camera...
        # ref, img = cap.read()

        # Code Running Status...
        # logger.info("Cam_1_alive-{}".format(datetime.datetime.now()))

                    # Checking blank images from live camera...
                    if img is None:
                        print("Camera Blank Image")
                        # logger.info("Blank image-{}".format(datetime.datetime.now()))
                        continue

                    # Model will SKIP every time 2 FRAMES...
                    # sec = sec + framerate
                    # if sec % 2 != 0:
                    #     continue

                    # Assigning color to the image
                    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

                    # font style
                    font = cv2.FONT_HERSHEY_PLAIN

                    # RESIZING image FRAME to display...
                    scale_percent = 40
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)

                    # Resize original image to display according to our requirement...
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    height, width, _ = img.shape
                    size = (width, height)

                    # Taking Height and Weight from resize image...
                    (W, H) = (None, None)
                    if W is None and H is None:

                        # Height and Weight taken from image, using images shape and index...
                        (H, W) = img.shape[:2]
                        # logger.info("Image Height and Weight has been captured")

                    # Convert image to blob format and send to vision model to detect object from the image...
                    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    ln = net.getUnconnectedOutLayersNames()
                    layerOutputs = net.forward(ln)
                    # logger.info("Image data has been convert into blob format")

                    # Assign empty list to append all detected details...
                    boxes = []
                    confidences = []
                    classIDs = []

                    # Running for loop into the layer output which came from blob format to know the score...
                    for output in layerOutputs:

                        for detection in output:
                            # logger.info("Inside for loop in layerouput")

                            # Taking score from the detection object...
                            scores = detection[5:]

                            # Taking max score from the detection object...
                            classID = np.argmax(scores)

                            confidence = scores[classID]
                            # logger.info("Print confidence score-{}".format(confidence))
                            # print(confidence)

                            # Getting the confidence score from the detected object...
                            if confidence >= 0.3:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)

                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append((float(confidence)))
                                classIDs.append(classID)

                                # Creating bounding box into the image...
                                box = detection[0:4] * np.array([W-2, H-7, W-2, H-7])

                                # box = detection[0:4] * np.array([W, H , W, H ])
                                (centerX, centerY, width, height) = box.astype("int")
                                x_a = int(centerX - (width / 2))
                                y_a = int(centerY - (height / 2))

                                # Appending box, confidence and class...
                                boxes.append([x_a, y_a, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

                                # logger.info("Print score score-{}".format(scores))
                                # logger.info("Print classID score-{}".format(classID))
                                # logger.info("Print confidence score-{}".format(confidence))
                                # logger.info("Class, Confidence score and Boxes has been appended")

                    # Finally here we can come to know Whether model detected any object or not...
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
                    # logger.info("Lenght of idxs-{}".format(idxs))

                    # If len of indexes is greater then 0 means object detected...
                    if len(idxs) == 0:
                        terminate += 1

                    # If length of idex > then o that means model has detect something...
                    if len(idxs) > 0:
                        terminate = 0

                        # Flatten the data which model has been detected...
                        for i in idxs.flatten():
                            # print("Object Name-{}".format(classes[classIDs[i]]))

                            # Creating bounding box into the code images from the wagon...
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])

                            color = [int(c) for c in colors[classIDs[i]]]
                            cv2.rectangle(img, (x, y), (x + 1 + w + 1, y + 3 + h + 2), color, 2)

                            # cv2.rectangle(img, (x, y), (x  + w , y + h ), color, 2)
                            # print(mapping)
                            # logger.info("running for loop into blob format")
                            # logger.info("mapping value-{}".format(mapping))

                            # check previously if any code is mapped with IR data. If mapped then code will not go inside the if condition, it will go directly else to continue...
                            if mapping != 1:

                                # If our model detected code then we need to perform below steps....
                                if "code" == classes[classIDs[i]]:
                                    # logger.info("Code has been detected from wagon")

                                    couplingcount = 0
                                    # Wagon_number += 1
                                    # Croping code image for OCR
                                    # print("code Detected")
                                    img_crop = img[y - 5:y - 5 + h + 15, x - 5:x - 5 + w + 15]

                                    # img_crop = img[y:y+ h, x:x + w]
                                    # cv2.putText(img_crop, "Code", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                    save_image += 1

                                    # Once crop done we need to check index shape value for reading easyocr...
                                    if img_crop.shape[1] != 0 and img_crop.shape[0] != 0:
                                        # Define color code for text
                                        color1 = (0, 0, 255)

                                        # Writing text in image
                                        cv2.putText(img, "Code", (x, y), font, 0.7, color1, 1, cv2.LINE_AA)

                                        # cv2.imwrite("/media/jsw/Data/Crop_Image/image_{}.jpg".format(save_image),img_crop)
                                        # print("Image saved")

                                        finalOCR_Result = Easy_OCR(img_crop)
                                        print(finalOCR_Result)
                                        #
                                        if finalOCR_Result:
                                            # print(codex)

                                            # if len(codex):
                                            # temp_list1 = re.findall('\d+', finalOCR_Result)
                                            # print("Easy OCR result-{}".format(temp_list1))
                                            allcode.append(finalOCR_Result)
                                            print("allcode-{}".format(allcode))

                                            try:
                                                print("Try to Connect SQL for Trigger")
                                                logger.info("Try to connect mssql server for Trigger")

                                                conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                                                                      'Server= 192.168.1.147;'
                                                                      # 'Server=DESKTOP-N33MAFB, 1433;'
                                                                      'Database=RMHS_5MT_rakeDetails;'
                                                                      'UID=sa;'
                                                                      'PWD=Password@123;'
                                                                      'MARS_connection=yes;')

                                                cursor = conn.cursor()
                                                print("SQL Connection successful for Trigger")

                                                query = "SELECT Message FROM Wagon_PlacedDetails;"
                                                cursor.execute(query)
                                                myresult = cursor.fetchall()
                                                value = []

                                                for k in myresult:
                                                    value.append(k)

                                                    # for j in value:
                                                    #     if mapping1 == 0:
                                                    if value[0][0] == "New Wagon Placed":
                                                        print("Condition matching for insert data")

                                                        # Taking maximum value from list and append to insertdta list
                                                        maxc = max(allcode, key=allcode.count)

                                                        if maxc not in codelist:

                                                            # print("maxc value -{}".format(maxc))
                                                            # print("maxc Lenght -{}".format(len(maxc)))
                                                            ct = datetime.datetime.now()
                                                            Rakeid = "Checking"
                                                            insertdata.append(ct)
                                                            insertdata.append(Rakeid)
                                                            insertdata.append(maxc)

                                                            cv2.imwrite("/home/jsw/Desktop/image/image_{}.jpg".format(save_image),img_crop)


                                                            # Append code list for reference
                                                            codelist.append(maxc)

                                                        # data - datetime.datetime(2021, 7, 6, 18, 20, 55, 877824),['22091354229']]

                                                            print("data-{}".format(insertdata))
                                                        # print("insert data 0-{}".format(insertdata[0]))
                                                        # print("insert data 1-{}".format(insertdata[1]))
                                                        # print("insert data 2-{}".format(insertdata[2]))
                                                        # print(insertdata[1])

                                                            # Calling SQl Function to insert data
                                                            insert_data(insertdata)
                                                            logger.info("insert data-{}".format(insertdata))
                                                            mapping = 1
                                                            # mapping1 = 1

                                                            if maxc:

                                                                # After insert data into  SQL delete triger table information
                                                                if __name__ == '__main__':
                                                                    Deletedata()

                                            except Exception as e:
                                                print("Theres an exception on SQL...-{}".format(e))

                                # If model has detect coupling then perform below steps...
                                else:

                                    if "coupling" == classes[classIDs[i]]:
                                        couplingcount += 1

                                        if couplingcount == 1 and mapping == 0:
                                            print("Going for codee not detected condition")

                                            # If not insert then insert xxxx
                                            id = "xxxxxxxxxxx"
                                            ct = datetime.datetime.now()
                                            Rakeid = "Checking"
                                            insertdata.append(ct)
                                            insertdata.append(Rakeid)

                                            # Insert Data into tabel
                                            insertdata.append(id)

                                            # After insert delete tiger table
                                            if __name__ == '__main__':
                                                Deletedata()

                                        mapping = 0
                                        insertdata = []
                                        allcode = []
                                        coupling += 1
                                        # Writing coupling in to the image
                                        # cv2.putText(img, "Coupling", (x, y), font, 0.8, (0,0,255), 1,cv2.LINE_AA)


                                        # logger.info("Coupling has been detected from Rake... restarting all variable")
                                        print("Coupling has been detected from Rake... restart all variable...........")

                            # If model has detect coupling then perform below steps...
                            else:

                                if "coupling" == classes[classIDs[i]]:
                                    couplingcount += 1

                                    if couplingcount == 1 and mapping == 0:
                                        print("Going for codee not detected condition")

                                        # If not insert then insert xxxx
                                        id = "xxxxxxxxxxx"
                                        ct = datetime.datetime.now()
                                        Rakeid = "Checking"
                                        insertdata.append(ct)
                                        insertdata.append(Rakeid)

                                        # Insert Data into table
                                        insertdata.append(id)

                                        # After insert delete tiger table
                                        if __name__ == '__main__':
                                            Deletedata()

                                    mapping = 0
                                    insertdata = []
                                    allcode = []
                                    coupling += 1

                                    # Writing coupling in to the image
                                    # cv2.putText(img, "Coupling", (x, y), font, 0.8, (0,0,255), 1,cv2.LINE_AA)

                                    couplingcount += 1
                                    # logger.info("Coupling has been detected from Rake... restarting all variable")
                                    print("Coupling has been detected from Rake... restart all variable...........")

                    # cv2.imshow("Video Image", img)
                    # cv2.waitKey(1)
                    # Get exception when any issue happen in above process...
    except Exception as e:
        print("Theres an exception...-{}".format(e))
        # logger.info("There is an exception-{}".format(e))