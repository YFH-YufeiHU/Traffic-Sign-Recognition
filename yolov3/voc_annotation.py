import xml.etree.ElementTree as ET
from os import getcwd
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import time

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["po","p3","p5","p6","p10","p11","p12","p19","p23","p26","p27"#11
,"pn","pg","pne","pr40","ph4","ph4.5","ph5","pl5"#8
,"pl20","pl30","pl40","pl50","pl60","pl70","pl80","pl100","pl120"#9
,"pm20","pm30","pm55","io","i2","i4","i5","ip","il60"#9
,"il80","il90","il100","wo","w13","w30","w55","w57","w59"]#9

classes_need_augmented = ["p3","p5","p6","p10","p12","p19","p23","p27"#8
,"pg","pr40","ph4","ph4.5","ph5","pl5"#6
,"pl20","pl30","pl70","pl120"#4
,"pm20","pm30","pm55","i2","ip"#5
,"il80","il90","il100","wo","w13","w30","w55","w59"]#8



classes_num = [0 for j in range(len(classes))]

def convert_annotation(year, image_id, list_file,is_writing,wd, year1, image_id1,is_augmented):
    flag = 0
    in_file = open('G:/github/Yolov3/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        classes_num[classes.index(cls)] += 1
        flag = 1
    if flag!= 0 and is_writing == True:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year1, image_id1))
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')
        flag = 0
    elif  flag != 0 and is_writing == False and is_augmented==True:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes_need_augmented or int(difficult) == 1:#If the class is not in the class that needs to be enhanced, then break out of the current loop
                continue
            cls_id = classes_need_augmented.index(cls)#Find out which indicators need to be strengthened
            #if classes_need_augmented[cls_id] ==0:
            #Convert the image of the found class that needs to be enhanced to the selected position
            img = cv2.imread('G:/github/Yolov3/VOCdevkit/VOC2007/JPEGImages/'+str(image_id1)+'.jpg')
            emptyImage = np.zeros(img.shape, np.uint8)
            emptyImage = img.copy()
            cv2.imwrite('./utils/ClassNeedBeAugmented/'+str(classes_need_augmented[cls_id])+'/'+str(image_id1)+'.jpg', emptyImage)
            shutil.copy('G:/github/Yolov3/VOCdevkit/VOC2007/Annotations/'+str(image_id1)+'.xml',\
                        './utils/ClassNeedBeAugmented/' + str(classes_need_augmented[cls_id]) + '/xml/')#Copy the corresponding xml file to the xml file of the enhancement class

def generate_files(is_writing,is_augmented):
    # wd = getcwd()
    wd = 'G:/github/Yolov3'
    list_file=[]
    sum_file=0
    for year, image_set in sets:
        image_ids = open('G:/github/Yolov3/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        if is_writing == True:
            list_file = open('%s_%s.txt'%(year, image_set), 'w')
            for image_id in image_ids:
                #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
                convert_annotation(year, image_id, list_file,is_writing,wd, year,image_id,is_augmented)
                sum_file+=1
            list_file.close()
        else:
            for image_id in tqdm(image_ids):
                convert_annotation(year, image_id, list_file,is_writing,wd, year,image_id,is_augmented)
                sum_file += 1
    print('the files numbers:',sum_file)
    return classes_num

if __name__=='__main__':
    classe_con = generate_files(True,False)
    classe_con1=0
    for i in range(len(classes)):
        classe_con1+=classe_con[i]
    print("statistics",classe_con1)
