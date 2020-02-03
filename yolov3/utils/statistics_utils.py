# -* encoding:utf-8 *-
import matplotlib.pyplot as plt
##########设置中文显示
from pylab import *
import pandas as pd
import voc_annotation
mpl.rcParams['font.sans-serif'] = ['SimHei']
font_size =11 # 字体大小
fig_size = (12, 6) # 图表大小
# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
mpl.rcParams['figure.figsize'] = fig_size


#######################第一种柱状图#################

data = voc_annotation.generate_files(False,False)
data_arg = argsort(data)
labels = voc_annotation.classes
index = np.arange(len(data))
#print(labels)
#print(data)
# 设置柱形图宽度
bar_width = 0.9

rects1=plt.bar(index, sorted(data),  tick_label=index, alpha=0.9, color='b', width=bar_width,align="center",edgecolor = 'white',)
plt.ylabel("Classes numbers")
# plt.xlabel(u"话题")
plt.title('Instance per category')
# 添加数据标签
def add_labels(rects):
    i=0
    for rect in rects:
        height = rect.get_height()
        height1=str(labels[data_arg[i]])+"/"+str(height)
        plt.text(rect.get_x() + rect.get_width() / 2, height+5, height1, ha='center', va='bottom',rotation=90,)
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')
        i+=1
add_labels(rects1)
plt.savefig("./statics.png")
plt.show()
