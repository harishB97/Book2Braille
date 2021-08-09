import numpy as np
import cv2
import copy


def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def bbArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


# only in pixels not in normalised co-ordinates
def bbIntersectionOverUnion(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou, interArea


def bbWithinAnother(boxA, boxB):
    _, yminA, _, ymaxA = boxA
    _, yminB, _, ymaxB = boxB
    if (yminB <= yminA <= ymaxB) and (yminB <= ymaxA <= ymaxB):
        return 1
    elif (yminA <= yminB <= ymaxA) and (yminA <= ymaxB <= ymaxA):
        return 2
    else:
        return 0


def expandBbs(bbs, image, labelMap):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxBlur = cv2.boxFilter(image, -1, (55, 20))
        boxBlur = cv2.normalize(boxBlur,  boxBlur, 0, 255, cv2.NORM_MINMAX)
        boxBlur = cv2.threshold(boxBlur, 240, 255, cv2.THRESH_BINARY_INV)[1]
        pixThreshold = {'text': 30, 'equation': 10}
        ht, wd = image.shape
        for bb in bbs:
            xmin, ymin, xmax, ymax = bb[2:]
            threshold = pixThreshold[labelMap[bb[0]]]
            for i in range((xmin+int((xmax-xmin)*(0.4))), 0, -1):
                pixAvg = np.average(boxBlur[ymin:ymax+1, i])
                if pixAvg < threshold:
                    xmin = i
                    break
            for i in range((xmax-int((xmax-xmin)*(0.4))), wd):
                pixAvg = np.average(boxBlur[ymin:ymax+1, i])
                if pixAvg < threshold:
                    xmax = i
                    break
            for i in range((ymin+int((ymax-ymin)*(0.4))), 0, -1):
                pixAvg = np.average(boxBlur[i, xmin:xmax+1])
                if pixAvg < threshold:
                    ymin = i
                    break
            for i in range((ymax-int((ymax-ymin)*(0.4))), ht):
                pixAvg = np.average(boxBlur[i, xmin:xmax+1])
                if pixAvg < threshold:
                    ymax = i
                    break
            bb[2:] = xmin, ymin, xmax, ymax
        return bbs


def bbCleanup(image, detectionsList, windowHt, windowStep, labelMap):
    bbs = []
    for i in range(len(detectionsList)):
        for detection in detectionsList[i]:
            if labelMap[detection.label] == "background":
                continue
            top = i * windowStep
            bb = frameNorm(image, (detection.xmin, (detection.ymin * windowHt) + top,
                    detection.xmax, (detection.ymax * windowHt) + top))
            bbs.append([detection.label, detection.confidence] + bb.tolist())
    cleanedBbs = copy.deepcopy(bbs)
    cleanedBbs = expandBbs(cleanedBbs, image, labelMap)
    cleanedBbs = [x+[0] for x in cleanedBbs]
    
    for i in range(len(cleanedBbs)):
            if cleanedBbs[i][-1] == 1: continue
            for j in range(len(cleanedBbs)):
                if i == j: continue
                if cleanedBbs[j][-1] == 1: continue

                i_class = cleanedBbs[i][0]
                i_conf = cleanedBbs[i][1]
                j_class = cleanedBbs[j][0]
                j_conf = cleanedBbs[j][1]

                iou, inter_area = bbIntersectionOverUnion(cleanedBbs[i][2:-1], cleanedBbs[j][2:-1])
                if iou > 0.7 and (i_class == j_class): # high overlap
                    if bbArea(cleanedBbs[i][2:-1]) > bbArea(cleanedBbs[j][2:-1]):
                        cleanedBbs[j][-1] = 1
                    else:
                        cleanedBbs[i][-1] = 1
                        break
                elif iou > 0.7 and (i_class != j_class): # high overlap
                    if i_conf > j_conf:
                        cleanedBbs[j][-1] = 1
                    else:
                        cleanedBbs[i][-1] = 1
                        break
                elif (inter_area/bbArea(cleanedBbs[i][2:-1])) > 0.7:
                    cleanedBbs[i][-1] = 1
                    break
                elif (inter_area/bbArea(cleanedBbs[j][2:-1])) > 0.7:
                    cleanedBbs[j][-1] = 1
                elif (inter_area/bbArea(cleanedBbs[j][2:-1])) > 0.7:
                    if i_conf > j_conf:
                        cleanedBbs[j][-1] = 1
                    else:
                        cleanedBbs[i][-1] = 1
                        break
                elif iou > 0:
                    box = bbWithinAnother(cleanedBbs[i][2:-1], cleanedBbs[j][2:-1])
                    if box == 1:
                        cleanedBbs[i][-1] = 1
                        break
                    elif box == 2:
                        cleanedBbs[j][-1] = 1
    
    cleanedBbs = [x[:-1] for x in cleanedBbs if x[-1] == 0]
    cleanedBbs = expandBbs(cleanedBbs, image, labelMap)

    return cleanedBbs, bbs
