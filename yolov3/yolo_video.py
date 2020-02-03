from yolo import YOLO
from yolo import detect_video

if __name__ == '__main__':
    video_path = './logs/000/video/v4.mp4'
    output_path = './detection_video/v4.mp4'
    detect_video(YOLO(), video_path, output_path)