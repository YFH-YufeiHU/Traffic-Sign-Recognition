# -*- coding: utf-8 -*-
import pymongo
import pandas as pd
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

def get_coll(name_db):
    client = pymongo.MongoClient('127.0.0.1', 27017)
    db = client.DataDb_detection
    if name_db=="Detection":
        user = db.Detection
        return user
    if name_db=="image":
        user = db.image
        return user
    return

class User(object):
    def __init__(self, name, advice):
        self.name = name
        self.advice = advice

    @staticmethod
    def query_users(name_db):
        users = get_coll(name_db).find()
        return users

    @staticmethod
    def get_advice(name_id):
        advice = get_coll('Detection').find({'name':str(name_id)})
        advice = list(advice)
        t1 = advice[0]  # 一条一条记录地读 (Series)
        t1 = pd.Series(t1)
        advice = t1[2]
        return advice

    @staticmethod
    def get_image(filename):
        img =get_coll('image').find_one({'filename':str(filename)})
        data = img['data']#img.read() # 获取图片数据
        image=Image.open(io.BytesIO(data))
        k = np.array(image)
        k = Image.fromarray(k)
        return k
if __name__ == '__main__':
    user1 = User.get_advice('pl40')
    user = User.get_image('pl40')
    fig = plt.figure(figsize=(12, 12))
    # fig.set_title("132")
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.imshow(user)
    plt.axis('off')
    plt.title("driver advice:"+user1)
    plt.show()
    # print(user)
# user['filename']
