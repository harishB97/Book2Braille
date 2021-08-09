import cv2
import depthai as dai
import numpy as np
import time
from preprocess_page import preprocess
from process_boundingbox import bbCleanup
from speech import say, listen
from image_to_content import image2Latex, image2Text
from content_to_braille import text2UEB, latex2Nemeth, writeToMasterBraille

labelMap = ["equation", "text"]
inputFrameShape = (416, 416)


def createLayoutParserPipeline(nnBlobPath):
    pipeline = dai.Pipeline()

    # Define a source - color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)
    
    # camera control
    controlIn = pipeline.createXLinkIn()
    controlIn.setStreamName('control')
    controlIn.out.link(camRgb.inputControl)

    nn_in = pipeline.createXLinkIn()
    nn_in.setStreamName("nn_in")

    # yolo network
    nn = pipeline.createYoloDetectionNetwork()
    nn.setBlobPath(nnBlobPath)
    nn.setConfidenceThreshold(0.25)
    nn.setNumClasses(2)
    nn.setCoordinateSize(4)
    nn.setAnchors(np.array([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]))
    nn.setAnchorMasks({"side39": np.array([0, 1, 2]), "side26": np.array([3, 4, 5]), "side13": np.array([6, 7, 8])})
    nn.setIouThreshold(0.5)
    
    nn_in.out.link(nn.input)

    nn_out = pipeline.createXLinkOut()
    nn_out.setStreamName("nn_out")
    nn.out.link(nn_out.input)

    return pipeline


def layoutParser(pipeline, brailleFile, brailleFileName):

    with dai.Device(pipeline) as device:
        device.startPipeline()
        camQue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        controlQueue = device.getInputQueue("control")
        nnInQue = device.getInputQueue(name="nn_in", maxSize=4, blocking=False)
        nnOutQue = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

        frame = None
        

        def addBoundingBoxes(frame, bbs):
            wt = 0.05
            c2 = (0, 0, 255)
            c1 = (0, 255, 0)
            for j in range(len(bbs)):
                label, _, xmin, ymin, xmax, ymax = [int(x) for x in bbs[j]]
                sub_img = frame[ymin:ymax+1, xmin:xmax+1]
                white = np.ones(sub_img.shape, dtype=np.uint8) * 255
                if labelMap[label] == 'equation':
                    cv2.rectangle(white, (0, 0), (white.shape[1], white.shape[0]), c1, -1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), c1, 2)
                elif labelMap[label] == 'text':
                    cv2.rectangle(white, (0, 0), (white.shape[1], white.shape[0]), c2, -1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), c2, 2)
                res = cv2.addWeighted(sub_img, 1-wt, white, wt, 1.0)
                frame[ymin:ymax+1, xmin:xmax+1] = res
                
            return frame


        def toPlanar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


        def runInference(nnInQue, nnOutQue, frame):
            buffer = dai.Buffer()
            windowHt = 0.5
            windowStep = 0.25
            windows = int((1 - windowHt) / windowStep) + 1
            detectionsList = []
            for i in range(windows):
                top = (i * windowStep) * frame.shape[0]
                bottom = top + (windowHt * frame.shape[0])
                inputFrame = frame[int(top) : int(bottom)].copy()
                buffer.setData(toPlanar(inputFrame, inputFrameShape))
                nnInQue.send(buffer)
                nnOut = nnOutQue.get()
                if nnOut is not None:
                    detections = nnOut.detections
                    detectionsList.append(detections)

            cleanedBbs, _ = bbCleanup(frame.copy(), detectionsList, windowHt, windowStep, labelMap)
            return cleanedBbs

        
        def parsePage(bbs):
            # sorting bounding boxes based on position (top to bottom) in page
            bbs = sorted(bbs, key=lambda bb: bb[3])
            parserOut = []
            for i, bb in enumerate(bbs):
                parserOut.append({
                    "coords": bb[2:],
                    "id": i+1,
                    "type": labelMap[bb[0]],
                    "content": None,
                    "braille": None})
            return parserOut


        pageNo = 1
        first = True
        time.sleep(5)
        while True:
            if first:  # trigger and disable auto-focus
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                ctrl.setAutoFocusTrigger()
                controlQueue.send(ctrl)
                time.sleep(5)

            try:
                if first:
                    audio = listen(speak="All set! Can begin once you say start")
                    first = False
                else:
                    audio = listen(speak="Please turn to next page and say start. Or to close the program please say stop")
            except:
                break
            if "start" in audio:
                say("capturing content")
            elif "stop" in audio:
                say("saving braille file. Please note down file name")
                time.sleep(5)
                say("File " + brailleFileName + " created.", rate=125)
                say("Thank you! Happy learning!")
                brailleFile.close()
                return
            else:
                continue
            
            time.sleep(2)
            camFrame = camQue.get()
            if camFrame is not None:
                frame = camFrame.getCvFrame()
                
                cv2.imshow("Book image", cv2.resize(frame, (1000, 640)))
                cv2.waitKey(1)

                leftPageUnc, rightPageUnc, leftPage, rightPage = preprocess(frame)
                
                cv2.imshow('Book image', cv2.hconcat([cv2.resize(leftPageUnc, (500, 640)), cv2.resize(rightPageUnc, (500, 640))]))
                cv2.waitKey(1)
                
                # Bounding box list -> [label, confidence, xmin, ymin, xmax, ymax]
                leftCleanedBbs = runInference(nnInQue, nnOutQue, leftPage.copy())
                rightCleanedBbs = runInference(nnInQue, nnOutQue, rightPage.copy())

                leftInferenceCleaned = addBoundingBoxes(leftPageUnc.copy(), leftCleanedBbs)
                rightInferenceCleaned = addBoundingBoxes(rightPageUnc.copy(), rightCleanedBbs)
                
                cv2.imshow('Book image', cv2.hconcat([cv2.resize(leftInferenceCleaned, (500, 640)), cv2.resize(rightInferenceCleaned, (500, 640))]))
                cv2.waitKey(1)

                parserOutLeft = parsePage(leftCleanedBbs)
                parserOutRight = parsePage(rightCleanedBbs)
                
                leftEqns = image2Latex([x for x in parserOutLeft if x['type'] == 'equation'], leftPage)
                say("left page equations done")
                leftText = image2Text([x for x in parserOutLeft if x['type'] == 'text'], leftPageUnc)
                say("left page text done")

                rightEqns = image2Latex([x for x in parserOutRight if x['type'] == 'equation'], rightPage)
                say("right page equations done")
                rightText = image2Text([x for x in parserOutRight if x['type'] == 'text'], rightPageUnc)
                say("right page text done")
                
                leftEqns = latex2Nemeth(leftEqns)
                leftText = text2UEB(leftText)

                rightEqns = latex2Nemeth(rightEqns)
                rightText = text2UEB(rightText)

                writeToMasterBraille(brailleFile, parserOutLeft, pageNo)
                pageNo += 1
                writeToMasterBraille(brailleFile, parserOutRight, pageNo)
                pageNo += 1

                say("Page completed. Braille transcription done")
