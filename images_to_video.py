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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
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

    # 获取视频文件的目录和文件名（不带扩展名）
    video_dir, video_filename = os.path.split(src_video_path)
    video_name, _ = os.path.splitext(video_filename)
    
    # 设置输出文件夹路径
    output_path = os.path.join(video_dir, f'{video_name}.MP4')

    images_to_video(src_video_path, output_path ,30.0)

    
