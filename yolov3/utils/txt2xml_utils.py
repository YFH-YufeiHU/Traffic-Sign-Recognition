from xml.dom.minidom import Document
import os
import os.path
from PIL import Image
import numpy as np
import time
import tqdm

def read_particular_line(dir,line):
    file = open(dir+'.txt', 'r')
    all_lines = file.readlines()
    file.close()
    return all_lines[line]

def txt_to_txt(dir_source,dir_to_path,num_less_40,img_path):
    # num_less_40 = []
    f = open(dir_source + '.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in range(len(lines)):
        line_content = read_particular_line(dir_source, line)
        line_content = line_content.split(" ")
        #print(line_content[0])
        #line_content_w = line_content[1].split("\n")
        if (str(line_content[0])=='pn')or (str(line_content[0])=='pne')or \
           (str(line_content[0])=='i5')or (str(line_content[0])=='p11')or \
           (str(line_content[0])=='pl40') or (str(line_content[0])=='pl50')or \
           (str(line_content[0]) == 'pl80' )or (str(line_content[0])=='po')or \
           (str(line_content[0]) == 'pl60' )or (str(line_content[0])=='pl100')or \
           (str(line_content[0]) == 'p26')or(str(line_content[0])=='io')or \
           (str(line_content[0]) == 'w57') or (str(line_content[0])=='i4')or \
           (str(line_content[0]) == 'p19')or(str(line_content[0])=='pl30')or \
           (str(line_content[0]) == 'p6') or(str(line_content[0])=='pl120')or \
           (str(line_content[0]) == 'il60') or (str(line_content[0])=='pm55')or \
           (str(line_content[0]) == 'pm30')or (str(line_content[0])=='pl5')or \
           (str(line_content[0]) == 'pl100') or (str(line_content[0]) == 'p26')or \
           (str(line_content[0]) == 'ph5') or (str(line_content[0]) == 'il90')or \
           (str(line_content[0]) == 'pg')or (str(line_content[0]) == 'pr40')or \
           (str(line_content[0]) == 'il90')or (str(line_content[0]) == 'p5')or \
           (str(line_content[0]) == 'ph4.5')or (str(line_content[0]) == 'ip')or \
            (str(line_content[0]) == 'p12')  or (str(line_content[0]) == 'p10')or \
            (str(line_content[0]) == 'w13')or (str(line_content[0]) == 'pl20') or\
            (str(line_content[0]) == 'pl70')or (str(line_content[0]) == 'i2')or \
            (str(line_content[0]) == 'p23')   :
            return

    for line in range(len(lines)):
        line_content = read_particular_line(dir_source, line)
        line_content = line_content.split(" ")
        if (int(line_content[1]) < 0 or int(line_content[1]) > 2048) or \
           (int(line_content[2]) < 0 or int(line_content[2]) > 2048) or \
           (int(line_content[3]) < 0 or int(line_content[3]) > 2048) or \
           (int(line_content[4]) < 0 or int(line_content[4]) > 2048) or \
           (abs(int(line_content[3]) - int(line_content[1])) < 40)   or \
           (abs(int(line_content[4]) - int(line_content[2])) < 40)   :
            continue
        if (abs(int(line_content[3])-int(line_content[1])) < 40)and (abs(int(line_content[4])-int(line_content[2])) < 40) :
            #num_less_40.append(1)
            num_less_40 += 1

            #conbine_=[num_less_40,str_num]
        #print(dir_to_path.split('./txt/')[1])
        img = Image.open(img_path+dir_to_path.split('./txt/')[1]+'.jpg')
        img.save('G:/github/Yolov3/keras-yolo3-master/utils/ClassNeedBeAugmented/all/aug/img_aug_need/'\
                 +dir_to_path.split('./txt/')[1]+'.jpg')
        file = open(dir_to_path + '.txt', 'a+', encoding='utf-8')
        file.write(line_content[0] + '\n' + line_content[1] + '\n' + line_content[2] + '\n' +
                   line_content[3] + '\n' + line_content[4] )
        file.close()
    return num_less_40

def writeXml(imgname, w, h, objbud, wxml):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("JPEGImages")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("TT100K Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("CHD VOC ")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode("3")
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for i in range(len(objbud) // 5):
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(objbud[i * 5+0])
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(objbud[i * 5 + 1])
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(objbud[i * 5 + 2])
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(objbud[i * 5 + 3])
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(objbud[i * 5 + 4])
        ymax.appendChild(ymax_txt)
        # threee-1#
        # threee#
    tmp = "G:/github/data/CCTSDB/VOCdevkit/VOC2007/"
    tempfile = tmp + "test.xml"
    with open(tempfile, "wb") as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

    rewrite = open(tempfile, "r")
    lines = rewrite.read().split('\n')
    newlines = lines[1:len(lines) - 1]

    fw = open(wxml, "w")
    for i in range(0, len(newlines)):
        fw.write(newlines[i] + '\n')

    fw.close()
    rewrite.close()
    os.remove(tempfile)
    return



if __name__ == "__main__":
    ann_path = "./txt/"
    img_path = "G:/github/Yolov3/keras-yolo3-master/utils/ClassNeedBeAugmented/all/aug/img_aug/"
    xml_path = "./xml/"
    dir_= "./txt/"
    dir = "./txt_txt/"
    files = os.listdir(dir)
    num_less_40 = 0
    for file in files:
        #print(file + "-->start!")
        num_less_40 = txt_to_txt(dir+os.path.splitext(file)[0] , dir_+os.path.splitext(file)[0] ,num_less_40,img_path)
        # if num_less_40==None:
        #     continue
        print(num_less_40)

    if not os.path.exists(xml_path):
        os.mkdir(xml_path)
    #打开之前转化的annotations中的txt文件
    # 判断源文件夹是否存在
    if (os.path.exists(ann_path)):
        files = os.listdir(ann_path)
        print(files)
        for file in files:
            print(file + "-->start!")
            img_name = os.path.splitext(file)[0] + '.jpg'
            print(img_name)
            fileimgpath = img_path + img_name
            print(fileimgpath)
            im = Image.open(fileimgpath)
            width = int(im.size[0])
            height = int(im.size[1])
            #提取标签txt文件
            filelabel = open(ann_path + file, "r+")
            lines = filelabel.read().split('\n')
            obj = lines[:len(lines) - 1]
            #存储xml文件
            filename = xml_path + os.path.splitext(file)[0] + '.xml'
            writeXml(img_name, width, height, obj, filename)
