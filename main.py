from pathlib import Path
from datetime import datetime
from spatial_detection import createSpatialDetectionPipeline, spatialDetection
from layout_parser import createLayoutParserPipeline, layoutParser

spatialBlobPath = str(Path(r"blob\mobilenet-ssd_openvino_2021.2_6shave.blob").resolve().absolute())
nnBlobPath = str(Path(r"blob\layout_parser.v2.yolov3_best_FP16_openvino_2021.3_5shave.blob").resolve().absolute())


def main():
    # spatial guidance to place the book correctly inside the RGB frame
    spatialPipeline = createSpatialDetectionPipeline(spatialBlobPath)
    spatialDetection(spatialPipeline)

    # parses the page layout into equation and text blocks
    brailleFileName = 'Braille doc ' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.nemeth'
    brailleFile = open(brailleFileName, 'w+', encoding="utf-8")
    layoutParserPipeline = createLayoutParserPipeline(nnBlobPath)
    layoutParser(layoutParserPipeline, brailleFile, brailleFileName)


main()
