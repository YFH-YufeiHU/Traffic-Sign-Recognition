import numpy as np
import matplotlib.pyplot as plt

class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = "train.txt"
    ##计算iou
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]#计算了一下需要聚类的box的数量
        k = self.cluster_number
        #将所有框信息的面积复制9份行=n,列=k,(ni,j)=S
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)#将面积重复9次
        box_area = np.reshape(box_area, (n, k))
        #随机选取的聚类框进行复制
        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])#行重复一次，列重复n次
        cluster_area = np.reshape(cluster_area, (n, k))
        #对应位置进行比较
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result
    #计算平均iou
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy
    #聚类算法
    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))#创建一个内部数据无实际意义的矩阵，大小为n*k
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters
    #聚类结果转化为txt
    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()
    #将filename装的图像地址读取并提取box的宽高信息
    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)#字典类型转化成矩阵
        f.close()
        print(result.shape)
        plt.figure()
        for i in range(3000):
            x=int(result[i][0])
            y=int(result[i][1])
            img=plt.scatter(x, y, alpha=0.6)
        plt.show()
        return result
    #k-mean算法的入口
    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]#按照第一行进行排序，宽信息
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "train.txt"##待聚类的文本信息
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
