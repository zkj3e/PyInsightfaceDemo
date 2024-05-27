import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


assert insightface.__version__>='0.7'

if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    img = cv2.imread('zta.jpg')
    faces = app.get(img)
    source_face = faces[0]
    print(source_face)
    res = cv2.imread('MJYS295_002.jpg')
    resFaces =  app.get(res)
    for face in resFaces:
        print('xxx')
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./zxa_swapped.jpg", res)


