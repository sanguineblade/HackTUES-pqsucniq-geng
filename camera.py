import cv2
import numpy as np

yoloConfig = "yolo-cfg/yolov4.cfg"
yoloWeights = "yolo-cfg/yolov4.weights"
yoloNames = "yolo-cfg/coco.names"

with open(yoloNames, "r") as f:
    classNames = f.read().strip().split("\n")

net = cv2.dnn.readNet(yoloNames, yoloNames)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layerNames = net.getLayerNames()
outputLayers = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)

    boxes = []
    confidences = []
    classIds = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.5 and classNames[classId] == "person":
                center_x, center_y, w, h = (
                    detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(classId)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
