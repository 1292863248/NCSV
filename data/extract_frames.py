import os
import cv2
from tqdm import tqdm


def extract_images(video_path, output_folder):
    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 新建文件夹
    output_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置帧间隔
    frame_interval = int(fps)
    # 逐帧提取并保存
    count = 0
    # 已提取的图片数
    img_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % frame_interval == 0:
                num = str(count // frame_interval).zfill(3)
                image_name = os.path.join(output_path, f"{num}.jpg")
                cv2.imwrite(image_name, frame)
                img_num += 1
            count += 1
            if img_num == 64:
                break
        else:
            break
    cap.release()


# 测试代码
if __name__ == '__main__':
    video_path = './video'  # 视频文件路径
    output_folder = './video_frames'  # 输出文件夹路径
    video_names = os.listdir(video_path)
    for video_name in tqdm(video_names):
        if video_name.split('.')[-1] == "mp4":
            extract_images(video_path + '/' + video_name, output_folder)
