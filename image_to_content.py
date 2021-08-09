import pytesseract
from mathpix import requestLatex
from datetime import datetime
from pathlib import Path
from time import sleep
import re

pytesseract.pytesseract.tesseract_cmd = str(Path("tesseract-ocr/tesseract.exe").resolve().absolute())


def tessOcr(img):
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,config="--psm 12")
    return " ".join(d["text"])


# For equations
def image2Latex(equations, image):
    for eqn in equations:
        xmin, ymin, xmax, ymax = eqn['coords']
        croppedImage = image[ymin:ymax+1, xmin:xmax+1].copy()
        imageName = "_".join([eqn['type'], str(eqn['id']), str(datetime.now())])
        attempts = 0
        while attempts < 5:
            try:
                attempts += 1
                rJson = requestLatex(croppedImage, imageName)
            except Exception as e:
                print(e)
                print('Latex request failed. Trying again')
                sleep(2)
                continue
            break
        else:
            print("Couldn't connect. Max attempts reached")
            continue

        if 'error' in rJson.keys():
            continue
        elif 'latex_styles' in rJson.keys():
            eqn['content'] = rJson["latex_styled"]
        else:
            eqn['content'] = rJson["text"]
            eqn['content'] = eqn['content'].replace('\\(', '')
            eqn['content'] = eqn['content'].replace('\\)', '')
            eqn['content'] = eqn['content'].replace('\\]', '')
            eqn['content'] = eqn['content'].replace('\\[', '')
        print(eqn['id'], "\t", eqn['content']) # added only for visual purpose
        return equations


# For texts
def image2Text(texts, image):
    for txt in texts:
        xmin, ymin, xmax, ymax = txt['coords']
        croppedImage = image[ymin:ymax+1, xmin:xmax+1].copy()
        txt['content'] = tessOcr(croppedImage)
        txt['content'] = re.sub(' +', ' ', txt['content'])
        txt['content'] = re.sub('\t+', ' ', txt['content'])
        print(txt['id'], "\t", txt['content']) # added only for visual purpose
    return texts
