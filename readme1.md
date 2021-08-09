# Transcription of Scientific literature to braille format using OAK-D

Ref: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
## Introduction:
---
This repo contains the training and impletation of converting Scientific literature to Braille format which then can be printed using a braille printer if available. 

## Dependencies:
---

- Opencv
- Tensorflow
- Depthai
- latex2nemeth.jar
- Tesseract

All dependencies can be installed using PIP

```Python
cd main
pip install -r requirements.txt
```

## Directory layout
---
```
main
│
├── blob\
│   ├── layout_parser.v1.yolov3_4000_FP16_openvino_2021.3_5shave.blob    
│   ├── layout_parser.v2.yolov3_best_FP16_openvino_2021.3_5shave.blob
│   ├── mobilenet-ssd_openvino_2021.2_6shave.blob
│   ├── ssd_mobilenet_v1_fpn_coco_v5_4442_5_shaves_FP16.blob
│   └── ssd_mobilenet_v1_fpn_coco_v5_4442_5_shaves_FP32.blob
│
├── latex2nemeth\
│   ├── encodings\
│   │   ├── nemeth.json
│   │   └── polytonic.json
│   │
│   ├── examples\
│   │   ├── mathpics.tex
│   │   ├── mathtest.tex
│   │   └── nemeth.json
│   │
│   ├── gpl-3.0.txt
│   ├── latex2nemeth
│   ├── latex2nemeth-v1.0.2.jar
│   └── README
│
├── tesseract-ocr\
│   ├── doc\
│   │   ├── AUTHORS
│   │   ├── LICENSE
│   │   └── README.md
│   │
│   ├── tessdata\
│   │   ├── configs\
│   │   │   ├── alto
│   │   │   ├── ambigs.train
│   │   │   ├── api_config
│   │   │   ├── bigram
│   │   │   ├── box.train
│   │   │   ├── box.train.stderr
│   │   │   ├── digits
│   │   │   ├── get.images
│   │   │   ├── hocr
│   │   │   ├── inter
│   │   │   ├── kannada
│   │   │   ├── linebox
│   │   │   ├── logfile
│   │   │   ├── lstm.train
│   │   │   ├── lstmbox
│   │   │   ├── lstmdebug
│   │   │   ├── makebox
│   │   │   ├── pdf
│   │   │   ├── quiet
│   │   │   ├── rebox
│   │   │   ├── strokewidth
│   │   │   ├── tsv
│   │   │   ├── txt
│   │   │   ├── unlv
│   │   │   └── wordstrbox
│   │   │
│   │   ├── tessconfigs\
│   │   │   ├── batch
│   │   │   ├── batch.nochop
│   │   │   ├── matdemo
│   │   │   ├── msdemo
│   │   │   ├── nobatch
│   │   │   └── segdemo
│   │   │
│   │   ├── eng.traineddata
│   │   ├── eng.user-patterns
│   │   ├── eng.user-words
│   │   ├── jaxb-api-2.3.1.jar
│   │   ├── osd.traineddata
│   │   ├── pdf.ttf
│   │   ├── piccolo2d-core-3.0.jar
│   │   ├── piccolo2d-extras-3.0.jar
│   │   └── ScrollView.jar
│   │
│   ├── ambiguous_words.exe
│   ├── classifier_tester.exe
│   ├── cntraining.exe
│   ├── combine_lang_model.exe
│   ├── combine_tessdata.exe
│   ├── dawg2wordlist.exe
│   ├── iconv.dll
│   ├── icudata57.dll
│   ├── icui18n57.dll
│   ├── icuuc57.dll
│   ├── libbz2-1.dll
│   ├── libcairo-2.dll
│   ├── libexpat-1.dll
│   ├── libffi-6.dll
│   ├── libfontconfig-1.dll
│   ├── libfreetype-6.dll
│   ├── libgcc_s_sjlj-1.dll
│   ├── libgif-7.dll
│   ├── libglib-2.0-0.dll
│   ├── libgobject-2.0-0.dll
│   ├── libgomp-1.dll
│   ├── libharfbuzz-0.dll
│   ├── libintl-8.dll
│   ├── libjbig-2.dll
│   ├── libjpeg-8.dll
│   ├── liblept-5.dll
│   ├── liblzma-5.dll
│   ├── libopenjp2.dll
│   ├── libpango-1.0-0.dll
│   ├── libpangocairo-1.0-0.dll
│   ├── libpangoft2-1.0-0.dll
│   ├── libpangowin32-1.0-0.dll
│   ├── libpcre-1.dll
│   ├── libpixman-1-0.dll
│   ├── libpng16-16.dll
│   ├── libstdc++-6.dll
│   ├── libtesseract-4.dll
│   ├── libtiff-5.dll
│   ├── libwebp-7.dll
│   ├── libwinpthread-1.dll
│   ├── lstmeval.exe
│   ├── lstmtraining.exe
│   ├── merge_unicharsets.exe
│   ├── mftraining.exe
│   ├── set_unicharset_properties.exe
│   ├── shapeclustering.exe
│   ├── tesseract-uninstall.exe
│   ├── tesseract.exe
│   ├── text2image.exe
│   ├── unicharset_extractor.exe
│   ├── wordlist2dawg.exe
│   └── zlib1.dll
│
├── braille_map.json           #json file for mapping text to braille text
├── content_to_braille.py      #Pyscript to convert text and image content to braille
├── image_to_content.py        #pyscript to convert image to Latex
├── layout_parser.py           
├── main.py
├── mathpix.py                 #Pyscript to convert convert image to latex through mathpix
├── preprocess_page.py         
├── process_boundingbox.py
├── speech.py
├── temp.tex
├── temp0.nemeth
└── template.aux
```

1. loren ipsum.
   - lorem ipsum
     - loremipsum 
     - 
2. Model was trained using
   [Transfer Learning of Tensorflow Mobilenet V2 image classification](https://tfhub.dev/s?module-type=image-classification&q=tf2) model

## Image collection for Training
---

`Images to train the model was created from the images taken from the Physics and Maths books [Automated Python script](https: book links)`

## Preprocessing of image
---
>Preprocessing of the captured image from the camera
![](media\pagetransformation.gif)

## Instruction to set physical setup
---
>Sample setup is shown

<img src="media\setup-oakd.jpg" width="500" height="400">



>1. OAK-D is place in the camera holder and connected to the host pc
>2. The holder is firmly attached to a table and Raised to 40cm from book
>3. Etc if any

## Performance metrics of YOLOV3 model

| Model |Trained Images| Input size | Accuracy| mAP | 
| --- | --- |--- | --- |--- |
|YOLOV3|3000| 300x300| ##.##% | ## |


## To run the model
---

```Python
cd main
python main.py
```
## Sample inference from the model
---
Consider the following image:

Text is highlighted in the Red
Math expressions are highlighted in green
<img src="media\inferences/left_page6.jpg" width="800" height="800">


## Output from the model
---
>Text portion of the book is dictated upon user instruction 
>Braille format of each page read is stored as .nemeth file in the /main directory which canbe printed using the Braille printer