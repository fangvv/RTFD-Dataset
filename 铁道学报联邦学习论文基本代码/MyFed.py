import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers,Sequential
from tensorflow import keras
import matplotlib.pyplot as plt
import math
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import time

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filter_num, (3,3), strides = 1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride!=1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride, padding='same'))
        else:
            self.downsample = lambda x:x
    
    def call(self, input, training = None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output
    
class ResNet(keras.Model):
    def __init__(self, layer_dims, channel = [64,64,128,256,512], num_classes = 1):
        #layer_dims形如[2,2,2,2]
        super(ResNet, self).__init__()
        self.channel = channel
        #最开始的7×7卷积和池化
        self.stem = Sequential([
            layers.Conv2D(self.channel[0], (7,7), strides=(2,2),padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')
        ])
        #block
        self.layer1 = self.build_resblock(self.channel[1], layer_dims[0])
        self.layer2 = self.build_resblock(self.channel[2], layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(self.channel[3], layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(self.channel[4], layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes,activation='sigmoid')
    
    def build_resblock(self, filter_num, blocks, stride = 1):
        res_blocks = Sequential()

        res_blocks.add(BasicBlock(filter_num, stride))
        '''
        一个block里面有2个 basicblock，也就是4层卷积
        按照ResNet18的结构，一共4个block，第一个全是stride=1，后面3个block，第一个basicblock里的第一层是stride=2的
        '''
        
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def call(self, input, training=None):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b,c]
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet([2, 2, 2, 2])
    


class myFed:
    def __init__(self,input_shape, X, Y, VX = None, VY = None, C = []):
        self.model = None
        self.x = X   #训练集
        self.y = Y   #标签
        self.vx = VX   #训练集
        self.vy = VY   #标签

        self.input_shapes = input_shape
        self.data_num = X.shape[0]  #训练集样本数目  
        self.SS = []    #保存量化的时候，各层权重的参数S
        self.ZZ = []    #保存量化的时候，各层权重的参数Z
        self.mtype = "ResNet18"   #网络类型  string
        self.history = None

    def change_sample(self,nX,nY):
        self.x = nX   #训练集
        self.y = nY   #标签

    def create_model(self, learning_rate = 0.001, losses = 'binary_crossentropy', channels = [64,64,128,256,512]):
        self.model = ResNet(layer_dims = [2,2,2,2], channel = channels)
        self.model.build(input_shape=self.input_shapes)
        self.model.summary()
        self.model.compile(optimizer=keras.optimizers.Adam(lr = learning_rate),
            
            loss = losses,
            metrics = ['accuracy'])

        return self.model

    def fit(self,batch = 20, epoch = 2, trainx = None,trainy = None, testx = None, testy = None, Generator = True):
        if trainx == None:
            trainx = self.x
        if trainy == None:
            trainy = self.y
        if testx == None:
            testx = self.vx
        if testy == None:
            testy = self.vy

        if Generator == True:
            train_datagen = ImageDataGenerator(
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

            test_datagen = ImageDataGenerator(
                                            rotation_range = 40,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True)

            train_generator = train_datagen.flow(trainx,trainy,batch_size = batch)
            validation_generator =  test_datagen.flow(testx,testy,batch_size = batch)

            self.history = self.model.fit(train_generator,
                                    validation_data=validation_generator,
                                    epochs=epoch,
                                    verbose=1)
        else:
            self.history = self.model.fit(trainx,trainy,
                                    validation_data=(testx,testy),
                                    epochs=epoch,
                                    verbose=1)
        '''
        log_dir= os.path.join('logs') #win10下的bug，
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
        self.model.fit(trainx, trainy, batch_size = batch, epochs = epoch, validation_split=0.1, callbacks=[tensorboard])
        '''

    def predict(self,x_pre):
        pre = self.model.predict(x_pre)
        return pre

    def quantization_R2Q(self, data_in):
        """
        quantization_R2Q  浮点到定点
        :param data_in: weight_float32
        :return: weight_int8
        """
        self.SS = []
        self.ZZ = []
        shape = data_in.shape
        Q = data_in
        num = data_in.size
        for i in range(num):
            Rmax = np.max(data_in[i])
            Rmin = (np.min(data_in[i]))

            if Rmax == Rmin and Rmax == 1:
                S = 1
                Z = 0
            else:
                S = (Rmax - Rmin)/255
                Z = 127 - Rmax/S

            self.SS.append(S)
            self.ZZ.append(Z)
            
            Q[i] = (data_in[i]/S + Z).astype(np.int8)

        return Q

    def quantization_Q2R(self, data_in, SS, ZZ):
        """
        quantization_R2Q 定点到浮点
        :param data_in: weight_int8
        :SS: S of each member of the weight array
        :ZZ: Z of each member of the weight array
        :return: weight_float32
        """
        shape = data_in.shape
        R = data_in
        num = data_in.size
        for i in range(num):

            S = SS[i]
            Z = ZZ[i]
            
            R[i] = ((data_in[i] - Z) * S).astype(np.float32)

        return R


    def channel_pruning(self, spares_rate, data_in):
        """
        通道剪枝
        将滤波器组在模型建立之后进行L1范数的排序，按spares_rate比率删掉小的
        channel pruning
        :param data_in: weight
        :param spares_rate: float
        :return: weight, model
        """

        new_size = []  #记录每个卷积层新的滤波器数目
        new_weight = []
        is_first = True
        first_new = 0  #第一次被剪滤波器之后的数目，之后的block依次×2, 用来帮助实现
                       #第一次按照比例剪掉相应数量后，后边block滤波器数增加，直接剪到上个block的2倍
        last_layer = 0 #上一次卷积层滤波器的数目
        cur_layer = 0 #当前卷积层滤波器的数目

        for i in range(len(data_in) - 2):
            if data_in[i].ndim == 4:
                if is_first:
                    is_first = False
                    last_layer = data_in[i].shape[3]
                elif data_in[i].shape[0] != 1:
                    
                    filter = data_in[i]
                    [kw, kh, r, c] = filter.shape
                    print(filter.shape)
                    #print(none_z)
                    filter = np.reshape(filter,[kw * kh, r, c])
                    filter_channelCut = np.zeros([kw * kh, z_sizes, c])
                    for j in range(z_sizes):
                        filter_channelCut[:,j,:] = filter[:, none_z[0,j], :]

                    filter_channelCut = np.reshape(filter_channelCut,[kw, kh, z_sizes, c])
                    data_in[i] = None
                    data_in[i] = filter_channelCut

                    if data_in[i + 8].shape[0] == 1:  #identity的对应删去的通道是由上一个block末尾卷积层删去的滤波器序号决定的
                        identity = data_in[i + 8]
                        [kw, kh, r, c] = identity.shape
                        print(filter.shape)
                        #print(none_z)
                        identity = np.reshape(identity,[kw * kh, r, c])
                        identity_channelCut = np.zeros([kw * kh, z_sizes, c])
                        for j in range(z_sizes):
                            identity_channelCut[:,j,:] = identity[:, none_z[0,j], :]

                        identity_channelCut = np.reshape(identity_channelCut,[kw, kh, z_sizes, c])
                        data_in[i + 8] = None
                        data_in[i + 8] = identity_channelCut


                filter = data_in[i]
                [kw, kh, r, c] = filter.shape
                cur_layer = c
                print(filter.shape)
                filter = np.reshape(filter,[kw * kh * r,c])
                L = []
                for j in range(c):
                    l1 = np.linalg.norm(filter[:,j])
                    L.append(l1)
                L = np.array(L)
                sort = np.sort(L)

                if last_layer == cur_layer and first_new == 0:
                    minl = sort[int(c * spares_rate)]
                    L[L < minl] = 0
                    none_z = np.nonzero(L)  #找到了需要保留的滤波器序号
                    new_size.append(len(L[none_z]))   #保存进新的滤波器数目列表里
                    z_sizes = len(L[none_z])
                    first_new = z_sizes

                elif last_layer == cur_layer:
                    minl = sort[len(L) - first_new]
                    L[L < minl] = 0
                    none_z = np.nonzero(L)  #找到了需要保留的滤波器序号
                    new_size.append(len(L[none_z]))   #保存进新的滤波器数目列表里
                    z_sizes = len(L[none_z])
                    if(z_sizes != first_new):
                        print("对不齐")
                    first_new = z_sizes

                else:
                    last_layer = len(L)
                    cur_layer = len(L)

                    minl = sort[len(L) - first_new * 2]
                    L[L < minl] = 0
                    none_z = np.nonzero(L)  #找到了需要保留的滤波器序号
                    new_size.append(len(L[none_z]))   #保存进新的滤波器数目列表里
                    z_sizes = len(L[none_z])
                    if(z_sizes != first_new * 2):
                        print("不是两倍")
                    first_new = z_sizes
                    
                

                new_filter = np.zeros([kw * kh * r,z_sizes])
                none_z = np.array(none_z)
                for j in range(z_sizes):
                    new_filter[:,j] = filter[:,none_z[0,j]]

                new_filter = np.reshape(new_filter,[kw, kh, r, z_sizes])
                new_weight.append(new_filter)  #保存进新的权重列表里
                print(filter.shape)
                #print(none_z)

            else:
                bias = data_in[i]
                new_bias = np.zeros([z_sizes])
                for j in range(z_sizes):
                    new_bias[j] = bias[none_z[0,j]]

                new_weight.append(new_bias)
                print(len(new_bias))
            
        new_res = self.create_model(channel=[new_size[0],new_size[1], new_size[5], new_size[10], new_size[15]])
        #new_res = ResNet([2,2,2,2],channel=[new_size[0],new_size[1], new_size[5], new_size[10], new_size[15]])
        weight = new_res.get_weights()
        for i in range(len(data_in) - 2):
            weight[i] = new_weight[i]

        new_res.set_weights(weight)
        print("新的各层滤波器数目:",new_size)

        return new_weight, new_res
        
'''
以下为读路径函数
'''
def read_path(path_name, images, labels):
    for dir_item in os.listdir(path_name):

        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path,images,labels)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = cv2.resize(image,dsize=(224,224))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image = np.reshape(image,(224,224,1))

                # 放开这个代码，可以看到resize_image()函数的实际调用效果
                # cv2.imwrite('1.jpg', image)

                images.append(image)
                #print(path_name)
                if path_name.split('/')[-1] == "Defective":
                    labels.append(0)
                else:
                    labels.append(1)

    return images, labels

def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))  # 均值
    std = np.std(X_train, axis=(0, 1, 2, 3))  # 标准差
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    
    return X_train, X_test




def Load_Image(train_path, test_path, normal=True):
    '''
    path_name: str 图片路径

    return : x_train, y_train, x_test, y_test  : numpy array
    '''
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_train,y_train = read_path(train_path,x_train,y_train)
    x_test,y_test = read_path(test_path,x_test,y_test)

    data_train = [(xx,yy) for xx,yy in zip(x_train, y_train)]

    np.random.shuffle(data_train)
    data_train = np.array(data_train)


    x_train = np.array([x for x,y in data_train],dtype = np.float32)
    y_train = np.array([y for x,y in data_train],dtype = np.float32)

    x_train = np.array(x_train[200:])
    y_train = np.array(y_train[200:])
    x_test = np.array(x_test[20:])
    y_test = np.array(y_test[20:])
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    if (normal):
        x_train, x_test = normalize(x_train, x_test)
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
        



'''
以下为本地训练函数
'''


def Local_Train(Name, sftp, x_train, y_train, x_test, y_test):
    '''
    Name: str  客户代号
    sftp: SSH变量
    x_train, y_train, x_test, y_test : numpy array
    '''

    client2 = myFed(input_shape = (None,224,224,1), X=x_train, Y=y_train, VX=x_test, VY=y_test)

    flag = True
    is_first = True

    is_newepoch = True   #是否为新的联邦轮次
    base_dir = os.getcwd()
    name = Name
    is_ok = False
    np.save('./is_ok.npy',[is_ok])
    while(flag):
        
        is_ok = np.load('./is_ok.npy')
        if(is_ok[0] != True):
            is_newepoch = True
            print("等待10s之后查询服务器是否完成聚合")
            strs = " "
            for i in range(10):
                time.sleep(1)
                strs+='#'
                print('\r[%-10s]'%(strs),end='')
            continue

        if(is_newepoch):
            
            '''
            获取服务器模型
            '''
            localpath='./server_weight/server_weight.npy'
            remotepath='/mnt/server_weight/server_weight.npy'
            print("downloading server_weight.npy")
            time_start=time.time()
            sftp.get(remotepath,localpath)
            time_end=time.time()
            print('time cost',time_end-time_start,'s')
            
            server_file = np.load('./server_weight/server_weight.npy')

            if(server_file[0] != True):
                flag = False
                break
            if(is_first):
                client2.channel = server_file[4]
                client2.create_model(channels = client2.channel)
                is_first = False
                
            
            client2.model.set_weights(client2.quantization_Q2R(server_file[1],server_file[2],server_file[3]))

            client2.fit(epoch = 10)
            client_weight = np.array(client2.model.get_weights())
            client_weight = client2.quantization_R2Q(client_weight)

            np.save('{}.npy'.format(name),[client2.data_num,client_weight, client2.SS, client2.ZZ])

            '''
            传输本地模型给服务器
            '''
            localpath = '{}.npy'.format(name)
            remotepath = '/mnt/client_weight/{}.npy'.format(name)

            print("uploading local model")
            time_start=time.time()
            sftp.put(localpath,remotepath)
            time_end=time.time()
            print('time cost',time_end-time_start,'s')

            is_newepoch = False
        else:
            continue