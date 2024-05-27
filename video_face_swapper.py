import cv2
import sys
import argparse
import os
import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
    parser.add_argument('face_image', type=str)
    return parser.parse_args()


def video_to_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    frame_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            break

        # 保存帧为图像文件
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"Extracted {frame_count} frames.")
    return fps

def images_to_video(image_folder, video_path, fps):
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 确保按顺序读取图片

    # 读取第一张图片以获取帧的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # 写入视频帧

    video.release()  # 释放视频写入对象
    print(f"Video saved to {video_path}")

if __name__ == '__main__':
    args = parse_args()

    src_video_path = args.video
    _, face_image_filename = os.path.split(args.face_image)
    face_image_filename, _ = os.path.splitext(face_image_filename)

    # 获取视频文件的目录和文件名（不带扩展名）
    video_dir, video_filename = os.path.split(src_video_path)
    video_name, _ = os.path.splitext(video_filename)
    
    # 设置输出文件夹路径
    img_output_folder = src_video_path


    fps = 30.0

    dst_video_path = os.path.join(video_dir, f'{video_name}_{face_image_filename}.mp4')

    ## swapper

    swap_output_folder = os.path.join(video_dir, f'{video_name}_{face_image_filename}')
    if not os.path.exists(swap_output_folder):
        os.makedirs(swap_output_folder)


    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    img = cv2.imread(args.face_image)
    faces = app.get(img)
    source_face = faces[0]

    count = 0
    for dirpath, _, filenames in os.walk(img_output_folder):
         for filename in filenames:
            if count % 10 == 0:
                start_time = time.time()
                image_path = os.path.join(dirpath, filename)
                src_image = cv2.imread(image_path)

                faces = app.get(src_image)
                for face in faces:
                    dst_iamge = swapper.get(src_image, face, source_face, paste_back=True)
                    cv2.imwrite(os.path.join(swap_output_folder, f'{filename}.jpg'), dst_iamge)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"代码运行时间: {execution_time} 秒")
            count += 1

    images_to_video(swap_output_folder, dst_video_path, fps)
