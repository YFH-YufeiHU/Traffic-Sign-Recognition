import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import tensorflow as tf
import keras
import os

def _main():
    #相关地址定义
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    #返回类别名称
    class_names = get_classes(classes_path)
    #返回anchors的
    anchors = get_anchors(anchors_path)
    #定义输入的格式
    input_shape = (416,416) # multiple of 32, hw
    #模型定义
    model = create_model(input_shape, anchors, len(class_names))
    #训练-train
    policy = 2
    train(policy, model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir)

def train(policy, model, annotation_path, input_shape, anchors, num_classes, log_dir='logs/'):
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, period=3,#3?
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1) # 当评价指标不在提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)# 当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练。
    # batch尺寸
    batch_size = 4
    # 这个表示，验证集占训练集的比例。建议划分大点。不然验证集的图片会很少。不利于验证集loss的计算
    val_split = 0.1
    #打开训练标签集
    with open(annotation_path) as f:
        lines = f.readlines()
    # 随机混淆train内容
    np.random.shuffle(lines)
    #从训练集中拿出一部分用来测试
    num_val = int(len(lines)*val_split)
    #用剩下的去真正训练
    num_train = len(lines) - num_val
    #输出训练样本集大小，以及测试集大小，以及batch_size

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if policy ==1:
        #create_model :freeze_body=true
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred},metrics= ['mae'])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=30,
                initial_epoch=0,
                callbacks=[checkpoint, logging])#tensorboard,以及断点续训同时进行
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if policy ==2:
        # create_model :freeze_body=true
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-5),loss={'yolo_loss': lambda y_true, y_pred: y_pred},metrics= ['mae'])  # recompile to apply the change
        print('Unfreeze all of the layers.')
        # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=132,
                            initial_epoch=81,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')
       # model.save(log_dir + 'model.h5')
    # Further training if needed.
    if policy ==3:
        # create_model :freeze_body=false
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), \
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['mae'])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors,
                                                                num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=46,  # epochs = 100，可以调小一点。设置的是20#epochs=100：实验可以少跑几轮看看效果
                            initial_epoch=28,
                            callbacks=[checkpoint, logging])  # tensorboard,以及断点续训同时进行
        model.save_weights(log_dir + 'trained_weights_fin.h5')
    if policy ==4:
        # create_model :freeze_body=false
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), \
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['mae'])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors,
                                                                num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,  # epochs = 100，可以调小一点。设置的是20#epochs=100：实验可以少跑几轮看看效果
                            initial_epoch=46,
                            callbacks=[checkpoint, logging, reduce_lr, early_stopping])  # tensorboard,以及断点续训同时进行
        model.save_weights(log_dir + 'trained_weights_fin.h5')
        model.save(log_dir + 'model.h5')
        # Further training if needed.
#输入类别地址，返回类别名称
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#输入anchor地址，返回9组anchor,即[[],[],[],[],[],[],[],[],[],,,[]]
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

#yoolo v3模型创建，不加载预训练模型，不冻结结构体，预训练模型所在的位置
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True,#2
            weights_path='./logs/000/trained_weights_final.h5'):
    K.clear_session() # get a new session
    #此处的Input函数，类似tensorflow定义tensor
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    #定义了三组tensor,也就是输出[?,13,13,3,85][],[?,52,52,3,(num_classes+5)]
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],num_anchors//3, num_classes+5)) for l in range(3)]
    #模型主体--默认参数下：y_true[l]的shape为（batch,H,W,3,num_classes+5)
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    #判断是否要重新加载模型
    if load_pretrained:
        if os.path.exists(weights_path):
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
        else:
            print("there is no corresponding model")
        if freeze_body:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = len(model_body.layers) - 3
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        #模型的损失函数-也是yolo v3的核心，自定义层[function, output_shape=None, mask=None, arguments=None]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    print('model_body.input: ', model_body.input)
    print('model.input: ', model.input)

    return model

#数据生成流？
def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

#数据生成器？
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    #随机混合数据，让训练的网络泛化能力更强
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
    _main()
