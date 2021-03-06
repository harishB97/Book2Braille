# Book2Braille

## Transcription of Scientific literature to braille format using OAK-D

This project was done as part of Opencv AI Competition 2021 (https://opencv.org/opencv-ai-competition-2021/)

### What is OAK-D?

OAK—D is a spatial AI powerhouse, capable of simultaneously running advanced neural networks while providing depth from two stereo cameras and color information from a single 4K camera in the center.

To buy visit page - https://store.opencv.ai/products/oak-d

### What is Book2Braille?

Book2Braille is a solution that is built to aid people with visual impairments in the field of science. The aim of the solution is to convert equations and texts from scientific books into Braille thereby making it accessible for visually impaired people.

<img src="media\system layout.jpg" width="768" height="432">

### **Follow the below link for in-depth details of the implementation and our motivation behind the project** 
### :point_right: **[Detailed report](https://drive.google.com/file/d/1fPOmdNyMbynQqpw4tFjVQ4-tJSE5wjyI/view?usp=sharing)**

### Setup
---
<img src="media\setup-oakd.jpg" width="500" height="400">

### How it works?
#### Step1: Image capturing and preprocessing
Capture image :arrow_right: Filter backgroud :arrow_right: Split pages :arrow_right: Uncurve the page :arrow_right: Adaptive threshold
<!-- <img src="media/7ccd7ec245f03e0b3195c3db9bd3d64a9d5a7f34.gif" width="800" height="450" /> -->
<img src="media/preprocessing_gif2.gif" width="800" height="450" />

#### Step2: Split image into three halves
<img src="media/page_split_in_three.jpg" width="768" height="432"/>

#### Step3: Feed each image to custom trained yoloV3 model to locate mathematical expressions and text blocks
<img src="media/bb_cleanup.jpg" width="768" height="500"/>

#### Step4: Extract content from text blocks and mathematical expressions using tesseract-OCR and MathPix respectively
####        Convert text to braille as per Universal English Braille standard and latex to braille using [Latex2Nemeth software](https://ctan.org/pkg/latex2nemeth?lang=en) 
<img src="media/text2braille.jpg" />
<img src="media/latex2braille.jpg"/>

### To run the program
---

```Python
cd Book2Braille
python main.py
```

### Dependencies (python packages):
---

- opencv
- depthai
- pytesseract
- scipy
- skimage
- speech_recognition
- pyttsx3

### Tools used:
---

- Custom trained yoloV3 model
- Tesseract
- [Mathpix](https://mathpix.com/)
- [Latex2Nemeth](https://ctan.org/pkg/latex2nemeth?lang=en)
