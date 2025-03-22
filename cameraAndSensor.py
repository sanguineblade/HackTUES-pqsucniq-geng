import os
import sys
import argparse
import glob
import time
import threading
import multiprocessing

import cv2
import numpy as np
from ultralytics import YOLO

from gpiozero import DistanceSensor 
import tkinter as tk 
from tkinter import font

stopEvent = multiprocessing.Event()   
measureProcess = None                  
timerThread = None                     
lastPersonTime = 0                     

def runDistanceMeasurementTk():
    sensor1 = DistanceSensor(echo=23, trigger=24, max_distance=10)
    sensor2 = DistanceSensor(echo=17, trigger=27, max_distance=10)
    distanceBound = 5

    window = tk.Tk()
    window.title("Distance Measurement")
    customFont = font.Font(size=30) 
    window.geometry("800x400") 

    distanceLabel1 = tk.Label(window, text="Distance: ", anchor='center', font=customFont)
    distanceLabel2 = tk.Label(window, text="Distance: ", anchor='center', font=customFont)
    distanceLabel3 = tk.Label(window, text="Posture: ", anchor='center', font=customFont)
    adviceLabel = tk.Label(window, text="", anchor='center', font=customFont)

    distanceLabel1.pack()
    distanceLabel2.pack() 
    distanceLabel3.pack()
    adviceLabel.pack()

    def isPostureCorrect(distance1, distance2):
        return abs(distance1 - distance2) < distanceBound

    def measureDistance():
        distance1 = int(sensor1.distance * 100)
        distance2 = int(sensor2.distance * 100)

        distanceLabel1.config(fg="blue", text="Distance: {} cm\n".format(distance1))
        distanceLabel2.config(fg="blue", text="Distance: {} cm\n".format(distance2))
        if isPostureCorrect(distance1, distance2):
            distanceLabel3.config(fg="green", text="Posture: good")
        else:
            distanceLabel3.config(fg="red", text="Posture: incorrect")

        window.after(100, measureDistance)

  
    def timerEndCallback():
        adviceLabel.config(text=adviceLabel.cget("text") + "\nDrink water, stand up and walk for abit.")

    window.after(3600 * 1000, timerEndCallback)

    measureDistance()
    window.mainloop()

def timerCallback():
    print("One hour timer reached.\ndrink water, stand up and walk for abit.")

def runYoloDetection():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                        required=True)
    parser.add_argument('--source', help='Image source: image file ("test.jpg"), folder ("test_dir"), video ("testvid.mp4"), or USB camera ("usb0")', 
                        required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold (example: "0.4")', default=0.5)
    parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480")', default=None)
    parser.add_argument('--record', help='Record results to "demo1.avi". Must specify --resolution.', action='store_true')
    args = parser.parse_args()

    modelPath = args.model
    imgSource = args.source
    minThresh = float(args.thresh)
    userRes = args.resolution
    record = args.record

    if not os.path.exists(modelPath):
        print('ERROR: Model path is invalid or model was not found.')
        sys.exit(0)

    model = YOLO(modelPath, task='detect')
    labels = model.names

    imgExtList = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
    vidExtList = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

    if os.path.isdir(imgSource):
        sourceType = 'folder'
    elif os.path.isfile(imgSource):
        _, ext = os.path.splitext(imgSource)
        if ext in imgExtList:
            sourceType = 'image'
        elif ext in vidExtList:
            sourceType = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(0)
    elif 'usb' in imgSource:
        sourceType = 'usb'
        usbIdx = int(imgSource[3:])
    elif 'picamera' in imgSource:
        sourceType = 'picamera'
        picamIdx = int(imgSource[8:])
    else:
        print(f'Input {imgSource} is invalid.')
        sys.exit(0)

    resize = False
    if userRes:
        resize = True
        resW, resH = map(int, userRes.split('x'))

    if record:
        if sourceType not in ['video', 'usb']:
            print('Recording works only for video and camera sources.')
            sys.exit(0)
        if not userRes:
            print('Please specify resolution for recording.')
            sys.exit(0)
        
        recordName = 'demo1.avi'
        recordFps = 30
        recorder = cv2.VideoWriter(recordName, cv2.VideoWriter_fourcc(*'MJPG'), recordFps, (resW, resH))

    if sourceType == 'image':
        imgsList = [imgSource]
    elif sourceType == 'folder':
        imgsList = [file for file in glob.glob(os.path.join(imgSource, '*'))
                    if os.path.splitext(file)[1] in imgExtList]
    elif sourceType in ['video', 'usb']:
        capArg = imgSource if sourceType == 'video' else usbIdx
        cap = cv2.VideoCapture(capArg)
        if userRes:
            cap.set(3, resW)
            cap.set(4, resH)
    elif sourceType == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cap.start()

    bboxColors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
                  (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
                  (98, 118, 150), (172, 176, 184)]

    avgFrameRate = 0
    frameRateBuffer = []
    fpsAvgLen = 200
    imgCount = 0

    global measureProcess, timerThread, lastPersonTime

    while True:
        tStart = time.perf_counter()

        if sourceType in ['image', 'folder']:
            if imgCount >= len(imgsList):
                print('All images processed. Exiting.')
                sys.exit(0)
            frame = cv2.imread(imgsList[imgCount])
            imgCount += 1
        elif sourceType == 'video':
            ret, frame = cap.read()
            if not ret:
                print('End of video file. Exiting.')
                break
        elif sourceType == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None:
                print('Camera error. Exiting.')
                break
        elif sourceType == 'picamera':
            frameBgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frameBgra), cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('Picamera error. Exiting.')
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        results = model(frame, verbose=False)
        detections = results[0].boxes

        objectCount = 0
        personDetected = False
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classIdx = int(detections[i].cls.item())
            className = labels[classIdx]
            conf = detections[i].conf.item()

            if className.lower() == 'person' and conf > minThresh:
                personDetected = True
                lastPersonTime = time.time()
                if measureProcess is None or not measureProcess.is_alive():
                    stopEvent.clear()
                    measureProcess = multiprocessing.Process(target=runDistanceMeasurementTk)
                    measureProcess.start()
                if timerThread is None:
                    timerThread = threading.Timer(3600, timerCallback)
                    timerThread.start()

            if conf > minThresh:
                color = bboxColors[classIdx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{className}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                labelYmin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, labelYmin - labelSize[1] - 10),
                              (xmin + labelSize[0], labelYmin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, labelYmin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                objectCount += 1

        if timerThread is not None and time.time() - lastPersonTime > 5:
            print("No person detected for over 5 seconds. Stopping distance measurement and timer.")
            stopEvent.set()
            if measureProcess is not None:
                measureProcess.terminate()
                measureProcess = None
            timerThread.cancel()
            timerThread = None

        if sourceType in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avgFrameRate:0.2f}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Number of objects: {objectCount}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('YOLO detection results', frame)
        if record:
            recorder.write(frame)

        key = cv2.waitKey(5) if sourceType not in ['image', 'folder'] else cv2.waitKey()
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            cv2.waitKey()
        elif key in [ord('p'), ord('P')]:
            cv2.imwrite('capture.png', frame)

        tStop = time.perf_counter()
        frameRateCalc = 1 / (tStop - tStart)
        if len(frameRateBuffer) >= fpsAvgLen:
            frameRateBuffer.pop(0)
        frameRateBuffer.append(frameRateCalc)
        avgFrameRate = np.mean(frameRateBuffer)

    print(f'Average pipeline FPS: {avgFrameRate:.2f}')
    if sourceType in ['video', 'usb']:
        cap.release()
    elif sourceType == 'picamera':
        cap.stop()
    if record:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runYoloDetection()
