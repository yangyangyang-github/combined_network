import torch
from torch import nn
from d2l import torch as d2l
from no_lvbo.li.net_2 import resnet18, resnet34, resnet50, resnet101, resnet152, Combine  # 目前只用了resnet18
import os
import pandas as pd
from no_lvbo.li.combineModelutils_2 import generate_map, seedVIG_Datasets
from torch.utils.data import DataLoader
import pandas as pd
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler


def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    # net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            
            y = y.to(device)            
            y_hat, states = net(X)
            loss = nn.MSELoss()
            metric.add(loss(y_hat, y)*X.shape[0]*X.shape[1], X.shape[0]*X.shape[1])
    return metric[0] / metric[1]

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            
      
if __name__ == '__main__':

    # # # data1 folder for SEED datasets
    label_dir = r'D:\Desktop\shangda\keyan\北京\学习\数据\自己数据\EEG_segments_2\CSVPerclos'
    data_dir = r'D:\Desktop\shangda\keyan\北京\学习\数据\自己数据\EEG_segments_2\CSV128Randint'
    test_participants = ['1'] # 定义测试集被试编码
    # 将整体数据按照被试分为训练集和测试集，如文件夹里已有map文件，会先删除再产生
    generate_map(data_dir, label_dir, test_participants)
'''
    train_map = r"D:\Desktop\shangda\keyan\北京\学习\拟合眼动数据\\no_lvbo\li\combine_mapfiles\train_data_map.csv"
    test_map = r"D:\Desktop\shangda\keyan\北京\学习\拟合眼动数据\\no_lvbo\li\combine_mapfiles\test_data_map.csv"
    
    # 网络训练基本参数设置
    batch_size = 32
    rnn_hidden_size = 64
    lr, num_epochs = 0.001, 100
    CNNnet = resnet50(classification=False) 
    CombNet = Combine(CNNnet, input_size=384*4, batch_first=True)
    # net.load_state_dict(torch.load('result\\ResNet18_end_randint_int0_900_acrossSub.params'))
    CombNet.apply(init_weights)
    device = d2l.try_gpu()
    print('training on', device) 
    CombNet.to(device)
    optimizer = torch.optim.SGD(CombNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    lr_schduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.05)  # default =0.07
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)

    loss = nn.MSELoss()
    test_l_best = float('inf')
    train_l = torch.zeros(num_epochs)
    test_l = torch.zeros(num_epochs)
    for epoch in range(num_epochs):
        
        # start_num = 0
        start_num = random.randrange(0,1024)
        sample_num = 8
        # time_num = 0
        time_num = random.randrange(0,sample_num)
        # 设置dataset_type，可以选择‘classification’或‘regression’
        dataset_type = 'regression'
        train_dataset = seedVIG_Datasets(train_map,start_num, time_num, sample_num, dataset_type)
        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_dataset = seedVIG_Datasets(test_map, start_num, time_num, sample_num,  dataset_type)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        print(f'train epoch {epoch}')
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(2)
        CombNet.train()
        for i, (X, y) in enumerate(train_iter):
           
            optimizer.zero_grad()
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            X = X.to(device)           
            y = y.to(device)
            
         
            y_hat, states = CombNet(X)
            y = y.view(y_hat.shape)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():

                metric.add(l*X.shape[0]*X.shape[1],  X.shape[0]*X.shape[1])

        lr_schduler.step()
        scheduler_warmup.step()
        train_l[epoch] = metric[0] / metric[1]
        test_l[epoch] = evaluate_accuracy_gpu(CombNet,test_iter, d2l.try_gpu())
        print(f'train Loss {train_l[epoch]}  'f'test Loss {test_l[epoch]}')
        if test_l[epoch] < test_l_best:
            test_l_best = test_l[epoch]
            torch.save(CombNet.state_dict(), 'best_combRes50_randint_lr1_bs32_acrossSub_run2.params')
    torch.save([train_l,test_l],'loss_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
    torch.save(CombNet.state_dict(), 'end_combRes50_randint_lr1_bs32_acrossSub_run2.params')
'''