import cv2
import base64
import requests
import json


defaultHeaders = {
    "app_id": "harimanog_gmail_com_b420ad_8196f6",
    "app_key": "e24f1e320fdff1bba2c5a7c14c36fb35792066292709b278c5d0944cf7478f26",
    "Content-type": "application/json",
}


def encodeImage(image, imageName):
    encoding = "png"
    pngEncoded = cv2.imencode(".png", image)[1].tostring()
    return "data:" + imageName + "/" + encoding + ";base64," + base64.b64encode(pngEncoded).decode()


def requestLatex(image, imageName):
    imageUri = encodeImage(image, imageName)
    data = {"src": imageUri}
    r = requests.post("https://api.mathpix.com/v3/text", data=json.dumps(data), headers=defaultHeaders)
    return json.loads(r.text)
    