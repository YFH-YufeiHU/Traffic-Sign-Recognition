from yolo import YOLO
from yolo import detect_video,detect_img_for_test
from models import User
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def read_particular_line(dir,line):
    file = open(dir+'.txt', 'r')
    all_lines = file.readlines()
    file.close()
    return all_lines[line]

def link_mongodb(dir_source):
    f = open(dir_source + '.txt', 'r')
    lines = f.readlines()
    f.close()
    if len(lines) ==1:
        plt.figure()
        line_content = read_particular_line(dir_source, 0)
        advice = User.get_advice(line_content.split(' ')[0])
        img = User.get_image(line_content.split(' ')[0])
        plt.imshow(img)
        plt.axis('off')
        plt.title("driver advice:" + advice)
    if len(lines) >1:
        for line in range(len(lines)):
            plt.figure()
            line_content = read_particular_line(dir_source, line)
            advice = User.get_advice(line_content.split(' ')[0])
            img = User.get_image(line_content.split(' ')[0])
            plt.imshow(img)
            plt.axis('off')
            plt.title("driver advice:" + advice)

if __name__=='__main__':
    # root = tk.Tk()
    # root.withdraw()
    #
    # file_path = filedialog.askopenfilename()
    root = tk.Tk()  # 创建一个Tkinter.Tk()实例
    root.withdraw()  # 将Tkinter.Tk()实例隐藏
    default_dir = r"文件路径"
    file_path = tk.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(default_dir)))
    dir_='./logs/000/backup_txt_map/'
    #图片检测
    img = detect_img_for_test(YOLO(),True,file_path)
    fig = plt.figure(figsize=(12, 12))
    plt.figure(1)
    plt.imshow(img)
    plt.axis('off')
    dir=file_path.split('/')[-1].split('.')[0]
    file = link_mongodb(dir_+dir)
    plt.show()
