# 打印人脸关键信息

import cv2
import numpy as np
from insightface.app import FaceAnalysis

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# 初始化FaceAnalysis应用
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# 读取图像
image_path = './images/xz1.jpeg'
img = cv2.imread(image_path)

# 进行人脸检测和识别
face = app.get(img)[0]

print("边界框坐标（bbox）:", face.bbox)
print("关键点坐标（kps）:", face.kps)
print("检测置信度分数（det_score）:", face.det_score)
print("68个关键点的3D坐标（landmark_3d_68）:", face.landmark_3d_68)
print("姿态信息（pose）:", face.pose)
print("106个关键点的2D坐标（landmark_2d_106）:", face.landmark_2d_106)
print("性别信息（gender）:", face.gender)
print("年龄信息（age）:", face.age)
print("特征向量， 用于后续的人脸识别和验证任务（embedding）:", face.embedding)

