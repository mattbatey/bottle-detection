from ultralytics import YOLO
import cv2
import math 
import numpy as np
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 800)


def calculate_orientation(x_c, y_x, w, h, angle):
    if w >= h:
        orientation = angle
    else:
        orientation = angle + 90
    return orientation

def getTopLeftPoint(points):
    
    highestPoint_val = [999999, 999999]
    highestPoint_idx = -1
    secondHighestPoint_val = [999999, 999999]
    secondHighestPoint_idx = -1

    for i, point in enumerate(points):
        if point[1] < highestPoint_val[1] or highestPoint_val[1] == 999999:
            # move original highest to second place
            secondHighestPoint_idx = highestPoint_idx
            secondHighestPoint_val = highestPoint_val
            
            # set new highest
            highestPoint_val = point
            highestPoint_idx = i
        elif point[1] < secondHighestPoint_val[1]:
            secondHighestPoint_idx = i
            secondHighestPoint_val = point

    # determine which is the most left point
    if highestPoint_val[0] < secondHighestPoint_val[0]: # highest point is most left
        return [highestPoint_val[0], highestPoint_val[1]]
    else:   # highest point is most right
        return [secondHighestPoint_val[0], secondHighestPoint_val[1]]
        

# model
model = YOLO("weights/best_obb.pt")

# object classes
'''classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]'''


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)


    results = model.predict(img, conf=0.4, stream=True)

    # coordinates
    for r in results:
        boxes = r.obb

        for box in boxes:
            # bounding box
            x, y, w, h, rot = box.xywhr[0]
            x, y, w, h, rot = int(x), int(y), int(w), int(h), rot # convert to int values

            orientation = calculate_orientation(x, y, w, h ,rot)
            rot_deg = int(np.rad2deg(rot))

            # put box in cam
            x1 = max(int(x - w / 2), 0)
            y1 = max(int(y - w / 2), 0)
            x2 = min(int(x + w / 2), img.shape[1])
            y2 = min(int(y + w / 2), img.shape[0])

            rect = ((x, y), (w, h), int(rot_deg))
            #cv2.rectangle(img, rec=rect, color=(255, 0, 255), thickness=3)
            rotatedBox = cv2.boxPoints(rect)
            rotatedBox = np.int32(rotatedBox)
            
            cv2.drawContours(img, [rotatedBox], 0, (0,0,255), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", model.model.names[cls])

            # object details

            org = getTopLeftPoint(rotatedBox)#[int(x - w / 2), int(y - h / 2)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            color = (255, 0, 0)
            thickness = 2

            label = f"{model.model.names[cls]}, {(confidence)*100:.0f}%, {w}, {h}"

            cv2.putText(img, label, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()