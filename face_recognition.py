import cv2
from insightface.app import FaceAnalysis
import numpy as np

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# 初始化FaceAnalysis应用
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载两张图像
image_path1 = './images/xz1.jpeg'
image_path2 = './images/xz2.jpeg'
img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)

# 人脸检测和特征提取
faces1 = app.get(img1)
faces2 = app.get(img2)

# 检查是否检测到了人脸
if len(faces1) == 0 or len(faces2) == 0:
    print("One or both images don't contain any faces.")
    exit()

# 提取第一个人脸的特征向量
embedding1 = faces1[0].embedding

# 提取第二个人脸的特征向量
embedding2 = faces2[0].embedding


# 打印相似度分数
cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print("余弦相似度:", cosine_similarity)
