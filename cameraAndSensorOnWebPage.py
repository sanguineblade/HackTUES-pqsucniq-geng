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
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2

try:
    from gpiozero import DistanceSensor
    gpioAvailable = True
except ImportError:
    print("WARN: gpiozero library not found or not running on RPi. Posture detection disabled.")
    gpioAvailable = False
except Exception as e:
    print(f"WARN: Error initializing GPIO: {e}. Posture detection disabled.")
    gpioAvailable = False

stopEvent = multiprocessing.Event()
measureProcess = None
lastPersonTime = 0
personCurrentlyDetected = False

app = Flask(__name__)
picam2 = None
latestAnnotatedFrame = None
frameLock = threading.Lock()

latestPostureData = {
    "distance1": None,
    "distance2": None,
    "status": "Not Running",
    "error": None,
    "timer_alert": False
}
postureDataLock = threading.Lock()
lastTimerTriggerTime = 0
TIMER_INTERVAL_SECONDS = 3600

def runDistanceMeasurement():
    global latestPostureData, postureDataLock, lastTimerTriggerTime, stopEvent

    if not gpioAvailable:
        print("Posture measurement process started but GPIO unavailable. Exiting process.")
        with postureDataLock:
            latestPostureData["status"] = "Error"
            latestPostureData["error"] = "GPIO unavailable"
        return

    sensor1, sensor2 = None, None
    try:
        sensor1 = DistanceSensor(echo=23, trigger=24, max_distance=10)
        sensor2 = DistanceSensor(echo=17, trigger=27, max_distance=10)
        distanceBound = 4.5
        print("Distance sensors initialized in background process.")

        while not stopEvent.is_set():
            currentTime = time.time()
            dist1M, dist2M = None, None
            dist1Cm, dist2Cm = None, None
            postureStatus = "Unknown"
            errorMsg = None

            try:
                dist1M = sensor1.distance
                dist2M = sensor2.distance
                dist1Cm = int(dist1M * 100) if dist1M is not None else float('inf')
                dist2Cm = int(dist2M * 100) if dist2M is not None else float('inf')

                if dist1Cm > sensor1.max_distance * 100 * 1.1: dist1Cm = float('inf')
                if dist2Cm > sensor2.max_distance * 100 * 1.1: dist2Cm = float('inf')

                if dist1Cm == float('inf') or dist2Cm == float('inf'):
                    postureStatus = "Range?"
                else:
                    if abs(dist1Cm - dist2Cm) < distanceBound:
                        postureStatus = "Good"
                    else:
                        postureStatus = "Incorrect"

            except Exception as e:
                print(f"Error reading sensors in background process: {e}")
                errorMsg = "Sensor Read Error"
                postureStatus = "Error"

            timerAlertActive = False

            if currentTime - lastTimerTriggerTime >= TIMER_INTERVAL_SECONDS:
                 print(f"Timer interval reached ({TIMER_INTERVAL_SECONDS}s). Setting alert.")
                 timerAlertActive = True

            with postureDataLock:
                latestPostureData["distance1"] = dist1Cm if dist1Cm != float('inf') else None
                latestPostureData["distance2"] = dist2Cm if dist2Cm != float('inf') else None
                latestPostureData["status"] = postureStatus
                latestPostureData["error"] = errorMsg
                latestPostureData["timer_alert"] = timerAlertActive

            time.sleep(0.5)

        print("Stop event received in posture process. Exiting loop.")

    except Exception as e:
        print(f"Fatal error in runDistanceMeasurement process: {e}")
        with postureDataLock:
            latestPostureData["status"] = "Error"
            latestPostureData["error"] = f"Process failed: {e}"
    finally:

        if sensor1:
            sensor1.close()
        if sensor2:
            sensor2.close()

        with postureDataLock:
            latestPostureData["status"] = "Stopped"
            latestPostureData["timer_alert"] = False
        print("Distance measurement process finished and cleaned up.")

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Cam & Posture</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }
        #video_stream { max-width: 90%; height: auto; border: 1px solid black; }
        #data_container { margin-top: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }
        .status-good { color: green; font-weight: bold; }
        .status-incorrect { color: red; font-weight: bold; }
        .status-range { color: orange; font-weight: bold; }
        .status-error { color: red; font-weight: bold; }
        .status-unknown { color: grey; }
        .status-stopped { color: grey; font-style: italic; }
        .status-running { color: blue; }
        #timer_alert_div { margin-top: 15px; padding: 10px; background-color: #ffecb3; border: 1px solid #ffc107; border-radius: 5px; display: none; /* Hidden by default */ }
    </style>
</head>
<body>
    <h1>Live Feed & Posture Monitor</h1>
    <img id="video_stream" src="{{ url_for('videoFeed') }}" />

    <div id="data_container">
        <h2>Posture Information</h2>
        <p>Sensor 1 Distance: <span id="dist1">--</span> cm</p>
        <p>Sensor 2 Distance: <span id="dist2">--</span> cm</p>
        <p>Posture Status: <span id="status">Loading...</span></p>
        <p>Error: <span id="error_msg">None</span></p>
        <div id="timer_alert_div">
            <p><strong>Reminder:</strong> Time to drink water, stand up, and walk around!</p>
            <button onclick="dismissTimerAlert()">Dismiss</button>
        </div>
    </div>

    <script>
        const dist1Elem = document.getElementById('dist1');
        const dist2Elem = document.getElementById('dist2');
        const statusElem = document.getElementById('status');
        const errorElem = document.getElementById('error_msg');
        const timerAlertDiv = document.getElementById('timer_alert_div');
        let timerAlertWasActive = false; // Track if we've shown the alert

        function updateData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    dist1Elem.textContent = data.distance1 !== null ? data.distance1 : 'N/A';
                    dist2Elem.textContent = data.distance2 !== null ? data.distance2 : 'N/A';
                    statusElem.textContent = data.status || 'Unknown';
                    statusElem.className = ''; // Clear previous classes
                    if (data.status) {
                        statusElem.classList.add('status-' + data.status.toLowerCase().replace('?', 'range')); // Handle "Range?"
                    } else {
                         statusElem.classList.add('status-unknown');
                    }
                    errorElem.textContent = data.error || 'None';
                    if (data.timer_alert && !timerAlertWasActive) {
                        timerAlertDiv.style.display = 'block';
                        timerAlertWasActive = true;
                    } else if (!data.timer_alert) {
                        timerAlertDiv.style.display = 'none';
                        timerAlertWasActive = false;
                    }
                    if (data.status === 'Stopped' || data.status === 'Error') {
                        if (data.status === 'Error') statusElem.classList.add('status-error');
                        if (data.distance1 === null) dist1Elem.textContent = '--';
                        if (data.distance2 === null) dist2Elem.textContent = '--';
                    }
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    statusElem.textContent = 'Connection Error';
                    statusElem.className = 'status-error';
                    dist1Elem.textContent = '--';
                    dist2Elem.textContent = '--';
                    errorElem.textContent = 'Failed to fetch';
                });
        }
        function dismissTimerAlert() {
             timerAlertDiv.style.display = 'none';
        }
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

def generateFrames():
    global latestAnnotatedFrame, frameLock
    while True:
        with frameLock:
            if latestAnnotatedFrame is None:

                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Waiting for stream...', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder)
                if not flag: continue
                frameToYield = bytearray(encodedImage)
            else:
                frameToEncode = latestAnnotatedFrame.copy()

                (flag, encodedImage) = cv2.imencode(".jpg", frameToEncode)

                if not flag:
                    continue
                frameToYield = bytearray(encodedImage)

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               frameToYield + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def videoFeed():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def getData():
    with postureDataLock:

        dataToSend = latestPostureData.copy()
    return jsonify(dataToSend)

def runYoloDetection():
    global measureProcess, lastPersonTime, personCurrentlyDetected, picam2, latestAnnotatedFrame, frameLock
    global latestPostureData, postureDataLock, lastTimerTriggerTime

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "yolov8n.pt")', required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold (example: "0.4")', default=0.5, type=float)
    parser.add_argument('--resolution', help='PiCamera resolution WxH (example: "640x480")', default="640x480")
    args = parser.parse_args()
    modelPath = args.model
    minThresh = args.thresh
    userRes = args.resolution

    if not os.path.exists(modelPath):
        print(f'ERROR: Model path "{modelPath}" is invalid or model was not found.')
        sys.exit(1)

    try:
        resW, resH = map(int, userRes.split('x'))
    except ValueError:
        print(f'ERROR: Invalid resolution format "{userRes}". Use WxH (e.g., "640x480").')
        sys.exit(1)

    try:
        print("Initializing PiCamera2...")
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2.0)
        print("PiCamera2 initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize PiCamera2: {e}")
        sys.exit(1)

    print(f"Loading YOLO model: {modelPath}")
    try:
        model = YOLO(modelPath, task='detect')
        labels = model.names
        print("YOLO model loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        if picam2: picam2.stop()
        sys.exit(1)

    bboxColors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
                  (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
                  (98, 118, 150), (172, 176, 184)]

    avgFrameRate = 0
    frameRateBuffer = []
    fpsAvgLen = 50

    print("Starting detection loop...")
    try:
        while True:
            tStart = time.perf_counter()

            frameBgra = picam2.capture_array()
            if frameBgra is None:
                print('ERROR: Failed to capture frame from PiCamera. Skipping frame.')
                time.sleep(0.5)
                continue

            frame = cv2.cvtColor(frameBgra, cv2.COLOR_BGRA2BGR)

            results = model(frame, verbose=False, conf=minThresh)
            detections = results[0].boxes

            objectCount = 0
            personDetectedThisFrame = False
            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classIdx = int(detections[i].cls.item())
                className = labels[classIdx]
                conf = detections[i].conf.item()

                if className.lower() == 'person':
                    personDetectedThisFrame = True
                    lastPersonTime = time.time()

                    if not personCurrentlyDetected:
                         print("Person detected. Resetting presence timer and potentially starting measurement.")

                         lastTimerTriggerTime = time.time()
                         with postureDataLock:
                             latestPostureData["timer_alert"] = False

                    personCurrentlyDetected = True


                    if (measureProcess is None or not measureProcess.is_alive()) and gpioAvailable:
                        print("Starting posture measurement process...")
                        stopEvent.clear()
                        with postureDataLock:
                            latestPostureData = { "distance1": None, "distance2": None, "status": "Starting...", "error": None, "timer_alert": False }
                        measureProcess = multiprocessing.Process(target=runDistanceMeasurement, daemon=True)
                        measureProcess.start()

                color = bboxColors[classIdx % len(bboxColors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{className}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                labelYmin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, labelYmin - labelSize[1] - 10),
                              (xmin + labelSize[0], labelYmin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, labelYmin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                objectCount += 1

            if personCurrentlyDetected and not personDetectedThisFrame:
                if (time.time() - lastPersonTime > 7):
                    print("No person detected for > 7 seconds. Stopping posture measurement.")
                    personCurrentlyDetected = False
                    if measureProcess is not None and measureProcess.is_alive():
                        stopEvent.set()
                        measureProcess.join(timeout=2.0)
                        if measureProcess.is_alive():
                            print("Posture process did not exit cleanly, terminating.")
                            measureProcess.terminate()
                        measureProcess = None
                        with postureDataLock:
                           latestPostureData["status"] = "Stopped"
                           latestPostureData["timer_alert"] = False

            tStop = time.perf_counter()
            frameRateCalc = 1.0 / (tStop - tStart) if (tStop - tStart) > 0 else 0
            frameRateBuffer.append(frameRateCalc)
            if len(frameRateBuffer) > fpsAvgLen:
                frameRateBuffer.pop(0)
            avgFrameRate = np.mean(frameRateBuffer) if frameRateBuffer else 0

            cv2.putText(frame, f'FPS: {avgFrameRate:.1f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Objects: {objectCount}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with frameLock:
                latestAnnotatedFrame = frame.copy()

    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting...")
    finally:
        print("Cleaning up resources...")
        if picam2:
            print("Stopping PiCamera2...")
            picam2.stop()
        if measureProcess is not None and measureProcess.is_alive():
            print("Stopping posture measurement process...")
            stopEvent.set()
            measureProcess.join(timeout=2.0)
            if measureProcess.is_alive(): measureProcess.terminate()
        print("Cleanup complete.")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("Starting Flask server in background thread (http://0.0.0.0:5000)...")

    flaskThread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flaskThread.start()
    runYoloDetection()

    print("Main program finished.")
    sys.exit(0)