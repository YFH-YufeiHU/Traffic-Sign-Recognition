import random
import numpy as np
import os
import json
from tqdm import tqdm
import time

def TTK1002txt_detection(labeled_images):
    classes = list()
    annotation = dict()
    objects = dict()
    img_num = []

    annotation['id'] = labeled_images['id']
    img_num = annotation['id']
    objects['objects'] = labeled_images['objects']
    for i in range(len( objects['objects'])):
        classes.append(objects['objects'][i]['category'])
        classes.append(objects['objects'][i]['bbox']['xmin'])
        classes.append(objects['objects'][i]['bbox']['ymin'])
        classes.append(objects['objects'][i]['bbox']['xmax'])
        classes.append(objects['objects'][i]['bbox']['ymax'])

    file_write_obj = open("./txt/"+str(img_num)+".txt", "a+")
    for var in range(len(classes)):
        file_write_obj.writelines(str(classes[var]))
        file_write_obj.write('\n')
    file_write_obj.close()
    return classes

if __name__ == '__main__':
    datadir = "./json_annotations"
    filedir = datadir + "/annotations.json"
    ids = open(datadir + "/train/ids.txt").read().splitlines()
    annos = json.loads(open(filedir).read())
    annos.keys()
    print(",".join(annos['types']))
   # imgid = random.sample(ids, 1)[0]
#    labeled_images = annos['imgs'][str(ids)]
    vector = [i for i in range(len(ids))]
    print('saving...')
    for i in tqdm(vector):
        labeled_images = annos['imgs'][str(ids[i])]
        #print(labeled_images)
        TTK1002txt_detection(labeled_images)
    print('ending...')