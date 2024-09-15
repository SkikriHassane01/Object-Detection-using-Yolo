# 1.import the necessary packages
#----------------------------------
from ultralytics import YOLO     #|
import cv2                       #|
import cvzone as cz              #|  This is a wrapper for OpenCV that makes drawing bounding boxes, labels, and more easier.
import math                      #|
#----------------------------------

# for image
#-----------------------------------#|
# path = "./cars.png"          #|
# image = cv2.imread(path)            #|
# model = YOLO('Weights/yolov8n.pt')  #|
# image = cv2.resize(image,(700,700)) #|
# results = model(image, show=True)   #|
# cv2.waitKey(0)                      #|
#-----------------------------------#|

# 2.opening a video or webcam or a simple image 

# for the webcam
#------------------------------------
# cap = cv2.VideoCapture(0)        #|
# cap.set(3, 1280)                 #|
# cap.set(4, 720)                  #|
#------------------------------------

# for video
#------------------------------------------------
cap = cv2.VideoCapture("./video4.mp4")  #|
#------------------------------------------------



# 3.Load YOLOv8 model
# there is nano, small, medium, large, xlarge models available in YOLOv8
# models ==> 'weights'

model = YOLO('./Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:

    success,img = cap.read() #reads each frame of the video in a loop
    results = model(img,stream =True) # the results will contain bounding boxes, confidence scores, and class predictions.
    # stream = True for real-time processing.

    for result in results:
        boxes = result.boxes

        # extract the bounding box and class information also confidence level for each class
        for box in boxes:

            ## 1.bounding box
            # x1, y1: The coordinates of the top-left corner of the bounding box.
            # x2, y2: The coordinates of the bottom-right corner of the bounding box.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            w, h = x2-x1, y2-y1
            cz.cornerRect(img, (x1, y1, w, h))

            # confidence score
            confidence = math.ceil(box.conf[0]*100)/100 # 2 decimal places

            # class name
            classIndex = int(box.cls[0])
            cz.putTextRect(img,f'{classNames[classIndex]}{confidence}',(max(0,x1),max(30,y1)), scale=1, thickness=1,offset=6)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



