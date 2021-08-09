import subprocess
import base64
import json
from pathlib import Path

brailleMap = json.load(open("braille_map.json",))


def getBraille(text):
    braille = ""
    for char in text:
        try:
            braille += brailleMap[char.lower()]
        except KeyError:
            braille += brailleMap[" "]
    return braille


def writeTex(eqns):
    texFile = open("temp.tex", "w")
    first = r"\documentclass[12pt, letterpaper, twoside]{article}" + "\n"
    second = r"\usepackage[utf8]{inputenc}" + "\n\n"
    begin = r"\begin{document}" + "\n"
    end = r"\end{document}" + "\n"

    with open("temp.tex", "r+") as texFile:
        for l in [first, second, begin]:
            texFile.write(l)
        for e in eqns:
            if e['content'] is None:
                continue
            texFile.write("\\begin{equation}\n" + e['content'] + "\\end{equation}\n")
        texFile.write(end)


def texFile2Nemeth():
    texFile = '"{}"'.format(str(Path("temp.tex").resolve().absolute()))
    auxFile = '"{}"'.format(str(Path("template.aux").resolve().absolute()))
    nemethJson = '"{}"'.format(str(Path("latex2nemeth/encodings/nemeth.json").resolve().absolute()))
    jarFile = '"{}"'.format(str(Path("latex2nemeth/latex2nemeth-v1.0.2.jar").resolve().absolute()))
    cmd = "java -jar " + jarFile + " " + texFile + " " + auxFile + " -e " + nemethJson
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc.communicate()


def readNemethFile():
    nemethFile = open("temp0.nemeth", "rb")
    nem = nemethFile.read()
    nem64 = base64.b64encode(nem)
    denem = base64.b64decode(nem64)
    eqnNemeth = denem.decode("utf-16").split("\n\n")
    return eqnNemeth


def latex2Nemeth(equations):
    opening, closing = "⠸⠩", "⠸⠱"
    writeTex(equations)
    texFile2Nemeth()
    eqnNemeth = readNemethFile()
    for i ,eqn in enumerate(equations):
        if eqn['content'] is None:
            continue
        eqn['braille'] = opening + eqnNemeth[i] + closing

    return equations


def text2UEB(texts):
    for txt in texts:
        txt['braille'] = getBraille(txt['content'])
    return texts


def writeToMasterBraille(brailleFile, parserOut, pageNo):
    brailleFile.write("Page no: " + str(pageNo) + '\n') # added only for visual purpose
    brailleFile.write(getBraille("Page no: " + str(pageNo)) + '\n')
    for x in parserOut:
        if x['braille'] is not None:
            print(x['id'], x['type'], x['braille'])
            brailleFile.write(x['type'] + '\n') # added only for visual purpose
            brailleFile.write(x['braille'] + '\n\n')
