import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from xml.dom.minidom import Document
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import time
ia.seed(1)

#读取原图像的bounding box坐标
def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(float(bndbox.find('xmin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymin = int(float(bndbox.find('ymin').text))
        ymax = int(float(bndbox.find('ymax').text))
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin,ymin,xmax,ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist

#传入目标变换后的bounding boxe坐标，将原坐标替换成新坐标并生成新的xml文件
# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))

def change_xml_list_annotation(root, image_id, new_target,saveroot,id):

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有object节点
        bndbox = object.find('bndbox')  # 子节点下节点bondbox的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_w30" + str(id) + '.xml'))

def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def load_image(path):
    """
    依据index对相应的样本图片进行加载，同时执行resize操作，并对图像进行归一化操作
    """
    img = cv2.imread(path)
    #img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    '''
     输入数组
     输出数组，支持原地运算
     range normalization模式的最小值
     range normalization模式的最大值
     NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用
     dtype为负数时，输出数组的type与输入数组的type相同；
    '''
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, -1)

    return img
if __name__ == "__main__":
    # 相关地址
    IMG_DIR = "./ClassNeedBeAugmented/w30/"# 输入的影像文件夹路径
    XML_DIR = "./ClassNeedBeAugmented/w30/xml/"  # 输入的XML文件夹路径

    AUG_XML_DIR ="./ClassNeedBeAugmented/all/w30/xml_aug/"  # 存储增强后的XML文件夹路径
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "./ClassNeedBeAugmented/all/w30/img_aug/"  # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 2 # 每张影像增强的数量

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 建立lambda表达式
    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []


    # 影像增强
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        sometimes(iaa.GaussianBlur(sigma=(0, 0.5))), # iaa.GaussianBlur(0.5),
        iaa.Affine(
           #translate_px={"x": 0, "y": 590},#平移
            #scale=(2.5, 2.5),#缩放
            translate_px={"x": 15, "y": 15},  # 平移
            scale=(0.9, 0.95),  # 缩放
            rotate =(-20, 20),#旋转
            #order=[0, 1],  # 使用最邻近差值或者双线性差值
            # cval=(0, 255),  # 全白全黑填充
            # mode=ia.ALL  # 定义填充图像外区域的方法
        ), # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
      # iaa.Crop(percent=(0, 0.1))
        # 这里沿袭我们上面提到的sometimes，对随机的一部分图像做crop操作
    ])
    #对bounding box 进行操作
    for root, sub_folders, files in os.walk(XML_DIR):
        for name in files:
            bndbox = read_xml_annotation(XML_DIR, name)
            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                # # 缩放比例
                # img.thumbnail((img.size[0] / 2, img.size[1] / 2))
                img = np.array(img)
                #image_aug = seq_det.augment_images([img])[0]
                # 目的实现增强的可视化
                # emptyImage = np.zeros(image_aug.shape, np.uint8)
                # emptyImage = image_aug.copy()
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                            int(bbs_aug.bounding_boxes[0].y1),
                                            int(bbs_aug.bounding_boxes[0].x2),
                                            int(bbs_aug.bounding_boxes[0].y2)])
                #     cv2.rectangle(emptyImage, (int(bbs_aug.bounding_boxes[0].x1), int(bbs_aug.bounding_boxes[0].y1)),\
                #                   (int(bbs_aug.bounding_boxes[0].x2), int(bbs_aug.bounding_boxes[0].y2)), (255, 0, 0), 2)
                # #可视化
                # plt.imshow(emptyImage)
                # plt.axis('off')
                # #plt.show()
                # plt.ion()
                # plt.pause(0.5)  # 显示秒数
                #plt.close()
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_w30" + str(epoch) + '.jpg')
                image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list,AUG_XML_DIR,epoch)
                print(str(name[:-4]) + "_aug_w30" + str(epoch) + '.jpg')
                new_bndbox_list = []
