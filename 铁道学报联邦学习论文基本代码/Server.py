import numpy as np
from MyFed import *


flag = True
loss = 0
old_loss = 0
is_first = True
is_ok = False   #是否聚合结束
is_start = False
losses = []
acces = []
np.save('/mnt/server_weight/is_ok.npy',[is_ok])
np.save('/mnt/server_weight/is_start.npy',[is_start])
epoch = 0
while(flag):
    
    
    
    if(is_first):
        print("等待足够的客户端加入")
        while(is_start == False):
            client_num = 0
            for dir_item in os.listdir('/mnt/client_require'):
                if dir_item.endswith('.npy'):
                    client_num = client_num + 1
         
            if(client_num == 2):
                is_start = True
                np.save('/mnt/server_weight/is_start.npy',[is_start])
                print("客户端数目已够，可以开始联邦")
                break
            
                
        
        '''
        初始化服务器
        '''
        server = myFed(input_shape = (None,224,224,1), X=x_train, Y=y_train)
        server.create_model()

        
        #server.fit(epoch=1)
        server_weight = np.array(server.model.get_weights())
        #print(server_weight.shape)

        '''
        进行通道剪枝
        '''
        _, new_model = server.channel_pruning(0.4,server_weight)
        w = new_model.get_weights()

        #进行fine_tune，稍微恢复一些剪枝后的参数精度
        #server.fit(epoch = 1)
        
        server_weight = np.array(server.model.get_weights())
        '''
        有量化
        '''
        server_wei2ght = server.quantization_R2Q(server_weight)
       
        print(server_weight.shape)
        is_train = True
        np.save('/mnt/server_weight/server_weight.npy',[is_train, server_weight, server.SS, server.ZZ, server.channel])
        is_ok = True
        
        np.save('/mnt/server_weight/is_ok.npy',[is_ok,epoch])
        #epoch = 1
        
        is_first = False

        #announce()
        
        time.sleep(60)
    else:
        ## 不允许再下载了
        is_ok = False
        np.save('/mnt/server_weight/is_ok.npy',[is_ok,epoch])
        #announce()
        
        client_num = 0   #客户总数
        total_num = 0   #客户持有的样本总数
        server_weight = server_weight * 0
        for dir_item in os.listdir('/mnt/client_weight'):
            if dir_item.endswith('.npy'):
                client_num = client_num + 1
                client_file = np.load('/mnt/client_weight/{}'.format(dir_item),allow_pickle = True)
                client_weight = client_file[1]
                '''
                有量化
                '''
                client_weight = server.quantization_Q2R(client_file[1],client_file[2],client_file[3])
                
                #print(client_weight)
                server_weight = np.add(server_weight, np.array(client_weight) * int(client_file[0]))
                total_num = total_num + client_file[0]
                print(total_num)
        
        server_weight = server_weight / total_num
        
        server.model.set_weights(server_weight)
        pre = server.predict(x_train)
        pre = np.reshape(pre,(pre.shape[0],))
        loss = tf.reduce_mean(keras.losses.binary_crossentropy(y_train, pre, from_logits=True))
        
        #pre = new_res.predict(x_train)

        for i in range(len(pre)):
            if(pre[i])>0.8:
                pre[i] = 1
            else:
                pre[i] = 0
        acc = accuracy_score(y_train,pre)
        print("联邦轮次",epoch,"loss值:",loss.numpy(),"  acc:",acc)
        losses.append(loss.numpy())
        acces.append(acc)
        epoch = epoch + 1
        
        if(np.abs(loss - old_loss) < 0.001 and acc > 0.8):
            flag = False
            is_train = False
            np.save('/mnt/server_weight/server_weight.npy',[is_train])
            is_ok = True
            np.save('/mnt/server_weight/is_ok.npy',[is_ok,epoch])
            
        
        
        else:
            '''
            有量化
            '''
            server_weight = server.quantization_R2Q(server_weight)
           
            
            np.save('/mnt/server_weight/server_weight.npy',[is_train, server_weight, server.SS, server.ZZ])
            
            ## 允许下载
            is_ok = True
            np.save('/mnt/server_weight/is_ok.npy',[is_ok,epoch])
            time.sleep(30)
            #announce()
    
