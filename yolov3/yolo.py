# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
from models import User
#改进策略
#在阈值为0.3的前提下，对比了V3和V2的测试效果之后

#用来存储预测结果的txt文件
predict_result = './logs/000/backup_txt_map/'
#数据集的val-set的图片
img_root_path = './logs/000/vol_images_set/'
#将预测结果的图片输出的路径
result_path = './detection/'

def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)

class YOLO(object):
    def __init__(self):
        self.anchors_path = './model_data/yolo_anchors.txt'  # Anchors
        self.model_path = './logs/000/ep123-loss6.567-val_loss6.543.h5'  # 模型文件
        self.classes_path = './model_data/voc_classes.txt'  # 类别文件

        # self.model_path = 'model_data/ep074-loss26.535-val_loss27.370.h5'  # 模型文件
        # self.classes_path = 'configs/wider_classes.txt'  # 类别文件

        self.score = 0.30
        #self.score = 0.60
        self.iou = 0.45
        #self.iou = 0.10
        self.class_names = self._get_class()  # 获取类别
        self.anchors = self._get_anchors()  # 获取anchor
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw

        self.colors = self.__get_colors(self.class_names)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def __get_colors(names):
        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(names), 1., 1.)
                      for x in range(len(names))]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 转换~
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)  # anchors的数量
        num_classes = len(self.class_names)  # 类别数

        #加载模型参数
        self.yolo_model = yolo_body(Input(shape=(416, 416, 3)), 3, num_classes)
        self.yolo_model.load_weights(model_path)  # 加载模型参数

        print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))

        # 根据检测参数，过滤框
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names),
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, img_path,is_imageDetection):#检测照片的位置
        dit_vedio=[]
        start = timer()  # 起始时间

        pic_filename = os.path.basename(img_path)
        portion = os.path.splitext(pic_filename)
        if portion[1] == '.jpg' or portion[1] == '.png':
            txt_result = predict_result + portion[0] + '.txt'
            print('txt_result的路径是：' + txt_result)

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print('detector size {}'.format(image_data.shape))
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 检测出的框
        dit_vedio.append(len(out_boxes))##################################

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 厚度
        if is_imageDetection ==True:
            with open(txt_result,'a')as new_f:
                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = self.class_names[c]  # 类别
                    box = out_boxes[i]  # 框
                    score = out_scores[i]  # 执行度

                    label = '{} {:.2f}'.format(predicted_class, score)  # 标签
                    draw = ImageDraw.Draw(image)  # 画图
                    label_size = draw.textsize(label, font)  # 标签文字

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))  # 边框
                    new_f.write(str(label) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')

                    if top - label_size[1] >= 0:  # 标签文字
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):  # 画框
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=self.colors[c])
                    draw.rectangle(  # 文字背景
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
                    del draw
            end = timer()
            print(end - start)  # 检测执行时间
            return image
        else:
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]  # 类别
                box = out_boxes[i]  # 框
                score = out_scores[i]  # 执行度

                dit_vedio.append(predicted_class)  ##############################
                dit_vedio.append(score)            ##############################

                label = '{} {:.2f}'.format(predicted_class, score)  # 标签
                draw = ImageDraw.Draw(image)  # 画图
                label_size = draw.textsize(label, font)  # 标签文字

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))  # 边框
                # new_f.write(str(label) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')

                if top - label_size[1] >= 0:  # 标签文字
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):  # 画框
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(  # 文字背景
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
                del draw
            end = timer()
            print(end - start)  # 检测执行时间
            dit_vedio.append((end - start))  ##############################
            return image,dit_vedio

    def detect_objects_of_image(self, img_path):
        image = Image.open(img_path)
        assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维
        # print('detector size {}'.format(image_data.shape))

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('out_boxes: {}'.format(out_boxes))
        # print('out_scores: {}'.format(out_scores))
        # print('out_classes: {}'.format(out_classes))

        img_size = image.size[0] * image.size[1]
        objects_line = self._filter_boxes(out_boxes, out_scores, out_classes, img_size)
        return objects_line

    def _filter_boxes(self, boxes, scores, classes, img_size):
        res_items = []
        for box, score, clazz in zip(boxes, scores, classes):
            top, left, bottom, right = box
            box_size = (bottom - top) * (right - left)
            rate = float(box_size) / float(img_size)
            clz_name = self.class_names[clazz]
            if rate > 0.05:
                res_items.append('{}-{:0.2f}'.format(clz_name, rate))
        res_line = ','.join(res_items)
        return res_line

    def close_session(self):
        self.sess.close()


def detect_img_for_test(yolo,is_imageDetection,image_path):
    if image_path!='':
        image = Image.open(image_path)
        filename = os.path.basename(image_path)
        print('filename:' + filename)
        r_image = yolo.detect_image(image, image_path, is_imageDetection)
        r_image.save(result_path + filename)
    # for img_path in iterbrowse(img_root_path):
    #     print('img_path的路径是：' + img_path)
    #     image = Image.open(img_path)
    #     filename = os.path.basename(img_path)
    #     print('filename:' + filename)
    #     r_image = yolo.detect_image(image, img_path,is_imageDetection)
    #     # r_image.show()  # 先显示，然后再保存
    #     r_image.save(result_path + filename)
    yolo.close_session()
    return r_image
def detect_video(yolo, video_path, output_path):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        b, g, r = cv2.split(frame)
        image = cv2.merge([r,g,b])
        image = Image.fromarray(image)
        image,dir_vedio = yolo.detect_image(image,result_path,False)
        image = np.asarray(image)
        b, g, r = cv2.split(image)
        result = cv2.merge([r, g, b])
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.50, color=(255, 255, 0), thickness=3)
        cv2.putText(result, text='the namber of dectection:'+str(dir_vedio[0]), org=(120, 980), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 255, 255), thickness=2)
        for i in range(int(dir_vedio[0])):
            cv2.putText(result, text='the class of dectection:' + str(dir_vedio[i*2+1]),
                        org=(120, 1015),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255,0), thickness=2)
            cv2.putText(result,  text= "confidence:" + str(dir_vedio[i*2 + 2]),
                        org=(120, 1050),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.putText(result, text='driver advice:' + str(User.get_advice(dir_vedio[i * 2 + 1])),
                        org=(370, 30+35*i),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 255), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

# if __name__=='__main__':
    #detect_img()
    # detect_img_for_test(YOLO(),True,image_path)