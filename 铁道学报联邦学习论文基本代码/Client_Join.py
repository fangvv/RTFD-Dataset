import numpy as np
from MyFed import *
import paramiko

join = "join"
name = "client1"
np.save('./join.npy', [join, name])
# path_name 为图片路径
train_path = "D:\\ChromeDownload\\ZIP\\Train"
test_path = "D:\\ChromeDownload\\ZIP\\Validation"

'''
加载图片
'''
x_train, y_train, x_test, y_test = Load_Image(train_path, test_path)



'''
transport = paramiko.Transport(('hz.matpool.com',28113))
transport.connect(username='root',password='97223280')
sftp = paramiko.SFTPClient.from_transport(transport)
localpath = './join.npy'
remotepath = './join.npy'
sftp.put(localpath,remotepath)
'''

is_start = False
while(is_start == False):
    
    start = np.load('is_start.npy')
    if(start[0] != True):
        print("等待5s之后查询服务器是否开始联邦训练")
        strs = " "
        for i in range(10):
            time.sleep(0.5)
            strs+='#'
            print('\r[%-10s]'%(strs),end='')
        continue
    else:
        is_start = True

'''
以下为开始运行本地训练
'''


if(is_start):
    Local_Train(name, sftp, x_train, y_train, x_test, y_test)