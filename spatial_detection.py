import depthai as dai
import cv2
import numpy as np
import math
import time
from speech import say

def createSpatialDetectionPipeline(nnPath):
    pipeline = dai.Pipeline()
    manip = pipeline.createImageManip()
    manip.initialConfig.setResize(300, 300)
    
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    manip.setKeepAspectRatio(False)

    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    spatialDetectionNetwork.setConfidenceThreshold(0.25)
    spatialDetectionNetwork.setBlobPath(nnPath)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    manip.out.link(spatialDetectionNetwork.input)

    nnOut = pipeline.createXLinkOut()
    nnOut.setStreamName("detections")
    spatialDetectionNetwork.out.link(nnOut.input)
    
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo.setOutputDepth(True)
    stereo.setConfidenceThreshold(50)
    stereo.setOutputRectified(True)

    stereo.rectifiedRight.link(manip.inputImage)

    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")
    stereo.rectifiedRight.link(xoutRight.input)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    
    return pipeline


def position(X1, Y1):
    if -40 < X1 < 40 and -15 <= Y1 <= 15:
        text = "Book is placed correct"
        instruction = "Book is placed correct, wait for some time"
        return True, text, instruction

    elif X1 > 40:
        dist = (abs(X1) - 40) * 0.1
        dist = dist if dist <= 5 else 5
        text = "Move the book " + str(math.ceil(dist)) + " cm left"
        instruction = "Please move the book " + str(math.ceil(dist)) + " centimeter left"
        return False, text, instruction

    elif X1 < -40:
        dist = (abs(X1) - 40) * 0.1
        dist = dist if dist <= 5 else 5
        text = "Move the book " + str(math.ceil(dist)) + " cm right"
        instruction = "Please move the book " + str(math.ceil(dist)) + " centimeter right"
        return False, text, instruction

    elif Y1 > 15:
        dist = (abs(Y1) - 15) * 0.1
        dist = dist if dist <= 5 else 5
        text = "Move the book " + str(math.ceil(dist)) + " cm down"
        instruction = "Please move the book " + str(math.ceil(dist)) + " centimeter down"
        return False, text, instruction

    elif Y1 < -15:
        dist = (abs(Y1) - 15) * 0.1
        dist = dist if dist <= 5 else 5
        text = "Move the book " + str(math.ceil(dist)) + " cm up"
        instruction = "Please move the book " + str(math.ceil(dist)) + " centimeter up"
        return False, text, instruction


def display(inFrame, text, detections, mono, fps):
    height, width = mono.shape
    mono = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR) 
    if inFrame:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    for detection in detections:
        x1, x2, y1, y2 = int(detection.xmin*width), int(detection.xmax*width), int(detection.ymin*height), int(detection.ymax*height)
        X = int(detection.spatialCoordinates.x)
        Y = int(detection.spatialCoordinates.y)
        cv2.putText(mono, f"X: {X} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        cv2.putText(mono, f"Y: {Y} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        cv2.rectangle(mono, (x1, y1), (x2, y2), color, 6)
    cv2.rectangle(mono, (0, height-70), (int(width*0.4), height-30), color, cv2.FILLED)
    cv2.putText(mono, text, (40, height-40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(mono, "NN fps: {:.2f}".format(fps), (2, mono.shape[0] - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255))

    cv2.imshow("rectified right", mono)
    cv2.waitKey(1)
    

def spatialDetection(pipeline):
    flipRectified = True
    X1, Y1 = 0, 0
    text = "Instruction"
    instruction = None

    startTime = time.monotonic()
    counter = 0

    with dai.Device(pipeline) as device:
        X = []
        Y = []
        device.startPipeline()
        previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        rectifiedRight = None
        detections = []

        inFrame = False

        while True:
            inRectified = previewQueue.get()
            det = detectionNNQueue.tryGet()
            
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            rectifiedRight = inRectified.getCvFrame()

            if det is not None:
                detections = det.detections
                detections = [x for x in detections if x.label in (5, 20)]
                if len(detections) >= 1:
                    detections = [sorted(detections, key=lambda x: x.confidence)[-1]]

            if flipRectified:
                rectifiedRight = cv2.flip(rectifiedRight, 1)

            for detection in detections:
                if flipRectified:
                    swap = detection.xmin
                    detection.xmin = 1 - detection.xmax
                    detection.xmax = 1 - swap

                if len(X) < 100:
                    X.append(detection.spatialCoordinates.x)
                    Y.append(detection.spatialCoordinates.y)
                else:
                    X1 = int(np.average(X))
                    Y1 = int(np.average(Y))
                    X = []
                    Y = []
                    if not inFrame:
                        inFrame, text, instruction = position(X1, Y1)
                        display(inFrame, text, detections, rectifiedRight, fps)
                        say(instruction)
                        continue
                
            if inFrame:
                cv2.destroyAllWindows()
                return
            
            display(inFrame, text, detections, rectifiedRight, fps)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
