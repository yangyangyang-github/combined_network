import torch
from no_lvbo.li.net_2 import resnet18, resnet34, resnet50, resnet101, resnet152, Combine
from no_lvbo.li.combineModelutils_2 import generate_map, seedVIG_Datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
def get_classify_label(Perclos):
    tired_threshold = 0.35
    drowsy_threshold = 0.7
    classify_label = np.repeat(2,Perclos.shape)
    awake_ind = Perclos <= tired_threshold
    classify_label[awake_ind] = 1
    drowsy_ind = Perclos >= drowsy_threshold
    classify_label[drowsy_ind] = 3
    return classify_label


label_dir = r'D:\Desktop\shangda\keyan\北京\学习\数据\自己数据\EEG_segments_2\CSVPerclos'
data_dir = r'D:\Desktop\shangda\keyan\北京\学习\数据\自己数据\EEG_segments_2\CSV128Randint'
test_participants = ['4'] # 定义测试集被试编码
# 将整体数据按照被试分为训练集和测试集，如文件夹里已有map文件，会先删除再产生
generate_map(data_dir, label_dir, test_participants)

test_map = r"D:\Desktop\shangda\keyan\北京\学习\拟合眼动数据\\no_lvbo\li\combine_mapfiles\test_data_map.csv"

start_num = 0
# start_num = random.randrange(0,1024)
time_num = 0
dataset_type = 'regression'
sample_num = 1
test_dataset = seedVIG_Datasets(test_map, start_num, time_num, sample_num,  dataset_type)
batch_size = 1
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda:0")

Y_label = []
Y_hat = []
with torch.no_grad():
    CNNnet = resnet50(classification=False) 
    CombNet = Combine(CNNnet, input_size=384*4, batch_first=True)
    CombNet.load_state_dict(torch.load('D:\Desktop\shangda\keyan\北京\学习\拟合眼动数据\\no_lvbo\li\\best_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
    CombNet.eval()
    CombNet.to(device)
    state_size = (1,batch_size,64)
    init_h = torch.zeros(state_size).to(device='cuda')
    init_c = torch.zeros(state_size).to(device='cuda')
    prev_states = (init_h, init_c)
    for X, y in test_iter:
        X = torch.as_tensor(X, dtype=torch.float).to(device)        
        y = y.reshape(-1)
        Y_label += y  
              
        y_hat, prev_states= CombNet(X,prev_states)
        y_hat = y_hat.reshape(-1)
        Y_hat += y_hat

    Y_label = torch.tensor(Y_label)
    Y_hat = torch.tensor(Y_hat)
    L1_loss = torch.nn.L1Loss()
    L2_loss = torch.nn.MSELoss()
    print(L2_loss(Y_label,Y_hat))
    print(L1_loss(Y_label,Y_hat))
    print(sum(get_classify_label(Y_label)==get_classify_label(Y_hat))/Y_label.shape)
    predict_cm = confusion_matrix(get_classify_label(Y_label),get_classify_label(Y_hat),labels=[1, 2, 3])
    print(predict_cm)
    
    # torch.save([Y_label,Y_hat],'regressionResult_randint_int0_900_acrossSub_p03.data1')
    x = torch.arange(Y_label.__len__())
    plt.plot(x, Y_label, label='Perclos')
    plt.plot(x, Y_hat, label='Predict')
    plt.legend()
    plt.show()

    conf_matrix = np.array(predict_cm)
    corrects = predict_cm.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数
    per_kinds_true = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数
    jing = corrects / per_kinds_true
    zhao = corrects / per_kinds
    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_dataset.__len__()))
    # print(conf_matrix)
    print("每种疲劳总个数：", per_kinds)
    print("每种疲劳预测正确的个数：", corrects)
    print("每种疲劳的识别精确率为：{0}".format([rate * 100 for rate in jing]))
    print("每种疲劳的识别召回率为：{0}".format([rate * 100 for rate in zhao]))
    print("每种疲劳的识别F1为：{0}".format([rate * 100 for rate in jing * zhao * 2 / (jing + zhao)]))

    print("rmse:", sqrt(mean_squared_error(Y_label, Y_hat)))
    Y_label_ave = Y_label.mean()
    Y_hat_ave = Y_hat.mean()
    fenzi = ((Y_label - Y_label_ave)*(Y_hat - Y_hat_ave)).sum()
    fenmu1 = ((Y_label - Y_label_ave)*(Y_label - Y_label_ave)).sum()
    fenmu2 = ((Y_hat - Y_hat_ave) * (Y_hat - Y_hat_ave)).sum()
    fenmu = sqrt(fenmu1*fenmu2)
    cor = fenzi/fenmu
    print("cor:", cor)