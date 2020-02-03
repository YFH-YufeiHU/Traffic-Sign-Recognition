###############finalVersion#################
detection		--存放图片检测的结果
detecttion_video	--存放检测的视频结果
font--
logs--backup_txt_map	--存放检测的信息文本txt
    --video		--存放待检测的视频
    --vol_images_set	--存放待检测的图像
    			--存放模型文件
model_data		--存放训练的样本类别以及先验框大小信息
utils			--存放json转txt程序代码
     			--数据增强前后的样本分布统计
     			--样本统计程序(statistics_utils)
     			--txt转xml程序代码
    			--xml转txt程序代码
     			--数据增强代码(XMLaug_utils)
yolo3			--存放yolov3的模型程序代码
     			--存放构建模型所使用的部分函数(utils)
convert			--暂未用到
kmeans			--聚类算法
models			--存放mongodb数据库的链接程序
train_policy		--存放训练策略程序
user_predict		--存放图像检测程序，图片的输入
voc_annotation		--根据voc格式的数据制作训练样本
yolo			--存放检测的接口（检测视频和图像底层函数）
yolo_video		--存放视频检测程序，视频输入
###############mAP-master#################
input--detection-results--检测的结果，需txt文本
     --ground-truth	--图像的标签文件，需txt文本
     --images-optional	--检测的图像（可选）
results			--mAP计算的结果
scripts			--文件格式转化程序
main			--mAP计算主程序
###############VOCdevkit#################
Annotations		--图像的标签文件(xml)
backup			--备份文件
ImageSets		--存放待训练的文本信息
JPEGImages		--存放图像数据集
test			--将数据集按照一定的比例转化成训练样本、验证集、测试集，相关信息存储在JPEGImages的Main文件夹下
###############mongodb#################
			--mongodb数据库环境
###############robo 3t#################
.exe			--mongodb数据库可视化软件