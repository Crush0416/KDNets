'''
------------KnHDR-----------
'''

import numpy as np
from KDNets import *
from tools import Class_AzimuthLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

#---------functions-------------
def target_candidate_searching(train_kd, train_fd, train_azimuth, train_label, azimuth_temp):
    #    判定候选目标范围，并筛选出符合条件的先验目标信息---精髓
    if azimuth_temp+5 >= 360:
        target_candidate_indices = ((train_azimuth >= azimuth_temp-5) | (train_azimuth <= azimuth_temp-355)).nonzero().squeeze()

    elif azimuth_temp-5 <= 0:
        target_candidate_indices = ((train_azimuth >= azimuth_temp + 355) | (train_azimuth <= azimuth_temp+5)).nonzero().squeeze()
        
    else:
        target_candidate_indices = ((train_azimuth >= azimuth_temp-5) & (train_azimuth <= azimuth_temp+5)).nonzero().squeeze()   #  indices 返回值格式似乎不对，待调试
    
    target_candidate_kd, target_candidate_fd, target_candidate_label = train_kd[target_candidate_indices], train_fd[target_candidate_indices], train_label[target_candidate_indices]
    
    return target_candidate_kd, target_candidate_fd, target_candidate_label

def prior_knowledge_searching(target_candidate_kd, target_candidate_fd, target_candidate_label, features, idx, vote=False):   #   待完善
    KLD = []
    features += 0.001              # 加一个偏置防止分子出现log(0)
    target_candidate_fd += 0.001  # 加一个偏差防止出现分母为0
    for target_i in target_candidate_fd:
        KLD_i1 = F.kl_div(features[idx].log(), target_i, reduction='sum').view(-1)       # 左KLD散度
        KLD_i2 = F.kl_div(target_i.log(), features[idx], reduction='sum').view(-1)         # 右KLD散度
        KLD_i = 1 * KLD_i1 + 0 * KLD_i2                                                # 对称KLD散度
        KLD.append(KLD_i)
        
        #  余弦相似性
        # similarity = F.cosine_similarity(features[idx].view(1,-1), target_i.view(1,-1), dim=1)
        #  欧氏距离
        # similarity = torch.dist(features[idx], target_i, p=2).view(-1)
        
        # KLD.append(similarity)
    KLD = torch.cat(KLD, dim=0)
    KLD_sort, indices = torch.sort(KLD, dim=0, descending=False)
    KLD_label = target_candidate_label[indices]
    if vote:
        elements, counts = torch.unique(KLD_label[0:5], return_counts=True, dim=0)    # 统计元素出现次数
        _, count_sort_indices = torch.sort(counts, dim=0, descending=True)            # 对次数进行降序排列
        vote_decision = elements[count_sort_indices[0]]
        vote_index = torch.nonzero(KLD_label == vote_decision).squeeze()                  # 查找vote类别的索引信息
        if vote_index[0] != 0:
            aa = 1   # 调试用
        prior_knowledge = target_candidate_kd[indices[vote_index[0]]]                          # 返回KLD值最小的先验概率分布

    else:
        prior_knowledge = target_candidate_kd[indices[0]]
    
    return prior_knowledge, KLD_sort, indices

def probability_transforamtion(output):
    h, w = output.size()
    max_value, _ = torch.max(output, 1)
    base_number = torch.pow(max_value, 1/max_value)    
    output_transform = torch.pow(base_number.view(h,1), output)
    sum = torch.sum(output_transform, 1).view(h,1)
    output = output_transform / sum
    
    return output   

#-------device configuration-----------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU IS TRUE')
    print('CUDA VERSION: {}'.format(torch.version.cuda))
else:
    print('CPU IS TRUE')

#   hyperparameters setting
path = './models/models_relu/class3/SSA/fullmodel_283Ep_0.9993Acc.pth'
batch_size = 100
num_classes = 3

#---------------DataLoader-------------------
train_dataset = scipy.io.loadmat('../../Datasets/Azimuth_MSTAR/data_SOC/class3/train/data_train_128.mat')
test_dataset  = scipy.io.loadmat('../../Datasets/Azimuth_MSTAR/data_SOC/class3/test/data_test_128.mat')

traindata_am = train_dataset['data_am']
traindata_azimuth = np.int16(train_dataset['azimuth']).squeeze()
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量

testdata_am = test_dataset['data_am']
testdata_azimuth = np.int16(test_dataset['azimuth']).squeeze()
testlabel = test_dataset['label'].squeeze()

train_dataset = MyDataset(img=traindata_am, azimuth=traindata_azimuth, label=trainlabel, transform=transforms.ToTensor())
test_dataset  = MyDataset(img=testdata_am, azimuth=testdata_azimuth, label=testlabel, transform=transforms.ToTensor())
print('train data size: {}'.format(train_dataset.img.shape[0]))
print('test data size: {}'.format(test_dataset.img.shape[0]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#-----------model loading----------------
model = torch.load(path)
criterion = nn.CrossEntropyLoss()

#   计算模型参数
para_num = sum([param.nelement() for param in model.parameters()])
print('Model Parameters: {:.4f} M'.format(para_num/1e6))

#    按照方位角保存训练集目标类别概率分布-P
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0
    probability_distribution = []
    feature_distribution = []
    attention = []
    azimuth = []
    label = []
    label_predict = []
    for image_batch, azimuth_batch, label_batch in train_loader:
        image_batch   = image_batch.to(device)
        azimuth_batch = azimuth_batch.to(device)
        label_batch   = label_batch.to(device)
        
        output, features, attn = model(image_batch, azimuth_batch)
        loss = criterion(output, label_batch)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_batch).sum().item()
        total += label_batch.size(0)
        
        # output = F.softmax(output, dim=-1)    #转化为概率
        output = probability_transforamtion(output)
        
        probability_distribution.append(output)
        feature_distribution.append(features)
        attention.append(attn)
        azimuth.append(azimuth_batch)
        label.append(label_batch)
        label_predict.append(predicted)
    print('Training set:\n correct number: {}, total number: {}, Accuracy: {}, Loss: {}'.format(correct, total, 100*correct/total, total_loss))
    probability_distribution = torch.cat(probability_distribution, dim=0)
    feature_distribution = torch.cat(feature_distribution, dim=0)
    attention = torch.cat(attention, dim=0)
    azimuth = torch.cat(azimuth, dim=0)
    label = torch.cat(label, dim=0)
    label_predict = torch.cat(label_predict, dim=0)
    
    matrix = confusion_matrix(label.data.cpu().numpy(), label_predict.data.cpu().numpy())
    print('-------------confusion matrix----------------\n', matrix)
    print('-------------attentions------------\n', attention[100,:,:])
    #     知识等级划分
    pd_sort, indices = torch.sort(probability_distribution, dim=1, descending=True)   #按降序排列概率分布
    pd_diff = pd_sort[:,0] - pd_sort[:,1]       # 按列计算每一个样本之间的概率差值
    
    explicit_knowledge_indices = (pd_diff >= 0.3).nonzero().squeeze()                           #  显著性知识
    general_knowledge_indices  = ((pd_diff >= 0.1) & (pd_diff<0.3)).nonzero().squeeze()         #  一般性知识
    fuzzy_knowledge_indices    = (pd_diff < 0.1).nonzero().squeeze()                            #  模糊性知识

    train_ek = probability_distribution[explicit_knowledge_indices]
    train_gk = probability_distribution[general_knowledge_indices]
    train_fk = probability_distribution[fuzzy_knowledge_indices]
    
    train_kd = probability_distribution
    train_fd = feature_distribution
    train_attn = attention
    train_azimuth = azimuth
    train_label = label
    
    #  保存train_kd, train_azimuth, train_label, train_ek, train_gk, train_fk
    # scipy.io.savemat('./results/prior_knowledge_distribution.mat', {'train_kd':train_kd, 'train_azimuth':train_azimuth,
                        # 'train_label':train_label, 'train_ek':train_ek, 'train_gk':train_gk, 'train_fk':train_fk,
                        # 'train_fd':train_fd, 'train_attn':train_attn})

#-------------基于知识等级划分的目标识别--------------    
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0

    explicit_knowledge_fusion, explicit_feature_fusion, explicit_knowledge_label, explicit_knowledge_azimuth = [], [], [], []
    general_knowledge_fusion, general_feature_fusion, general_knowledge_label, general_knowledge_azimuth = [], [], [], []
    fuzzy_knowledge_fusion, fuzzy_feature_fusion, fuzzy_knowledge_label, fuzzy_knowledge_azimuth = [], [], [], []
    
    general_probability = []
    general_features = []
    general_prior_knowledge = []
    general_target_candidate_kd = []
    general_target_candidate_fd = []
    general_target_candidate_label = []
    general_KLD_sort = []
    general_indices = []
    
    fuzzy_probablity = []
    fuzzy_features = []
    fuzzy_prior_knowledge = []
    fuzzy_target_candidate_kd = []
    fuzzy_target_candidate_fd = []
    fuzzy_target_candidate_label = []
    fuzzy_KLD_sort = []
    fuzzy_indices = []
    
    
    for image_batch, azimuth_batch, label_batch in test_loader:
        image_batch   = image_batch.to(device)
        azimuth_batch = azimuth_batch.to(device)
        label_batch   = label_batch.to(device)

        output, features, attn = model(image_batch, azimuth_batch)
        loss = criterion(output, label_batch)
        total_loss += loss.item()
        
        # output = F.softmax(output, dim=-1)    #转化为概率
        output = probability_transforamtion(output)
        output_sort, indices = torch.sort(output, dim=1, descending=True)   #按降序排列概率分布
        pd_diff = output_sort[:,0] - output_sort[:, 1]       # 按列计算每一个样本之间的概率差值
        
        explicit_knowledge_indices = (pd_diff >= 0.35).nonzero().squeeze()                           #  显著性知识
        general_knowledge_indices  = ((pd_diff >= 0.1) & (pd_diff < 0.35)).nonzero().squeeze()       #  一般性知识
        fuzzy_knowledge_indices    = (pd_diff < 0.1).nonzero().squeeze()                            #  模糊性知识
        
        if explicit_knowledge_indices.numel() > 0:
            explicit_knowledge = output[explicit_knowledge_indices].view(explicit_knowledge_indices.numel(), -1)
            features_explicit = features[explicit_knowledge_indices].view(explicit_knowledge_indices.numel(), -1)
            azimuth_explicit = azimuth_batch[explicit_knowledge_indices].view(-1)
            label_explicit = label_batch[explicit_knowledge_indices].view(-1)
            
            explicit_knowledge_fusion.append(explicit_knowledge)
            explicit_feature_fusion.append(features_explicit)
            explicit_knowledge_label.append(label_explicit)
            explicit_knowledge_azimuth.append(azimuth_explicit)
            
        if general_knowledge_indices.numel() > 0:
            general_knowledge = output[general_knowledge_indices].view(general_knowledge_indices.numel(), -1)
            features_general = features[general_knowledge_indices].view(general_knowledge_indices.numel(), -1)
            azimuth_general = azimuth_batch[general_knowledge_indices].view(-1)
            label_general = label_batch[general_knowledge_indices].view(-1)
            
            for idx, azimuth_temp in enumerate(azimuth_general):
                
                #    判定候选目标范围，并筛选出符合条件的先验目标信息
                target_candidate_kd, target_candidate_fd, target_candidate_label = target_candidate_searching(train_kd, train_fd, train_azimuth, train_label, azimuth_temp)
                #    从候选目标中找出最符合测试目标分布的先验知识
                prior_knowledge, KLD_sort, indices = prior_knowledge_searching(target_candidate_kd, target_candidate_fd, target_candidate_label, features_general, idx, vote=True)
                #    一般性知识融合
                namda = 0
                knowledge_fusion = namda*general_knowledge[idx] + (1-namda)*prior_knowledge
                general_knowledge_fusion.append(knowledge_fusion.view(-1, 10))
                
                #   调试start---
                general_target_candidate_kd.append(target_candidate_kd.data.cpu().numpy())
                general_target_candidate_fd.append(target_candidate_fd.data.cpu().numpy())
                general_target_candidate_label.append(target_candidate_label.data.cpu().numpy())
                general_prior_knowledge.append(prior_knowledge.data.cpu().numpy())
                general_KLD_sort.append(KLD_sort.data.cpu().numpy())
                general_indices.append(indices.data.cpu().numpy())
                #   调试end---
            
            general_knowledge_label.append(label_general)
            general_knowledge_azimuth.append(azimuth_general)
            general_features.append(features_general)
            general_probability.append(general_knowledge)
            
        if fuzzy_knowledge_indices.numel() > 0:
            fuzzy_knowledge = output[fuzzy_knowledge_indices].view(fuzzy_knowledge_indices.numel(), -1)
            features_fuzzy = features[fuzzy_knowledge_indices].view(fuzzy_knowledge_indices.numel(), -1)
            azimuth_fuzzy = azimuth_batch[fuzzy_knowledge_indices].view(-1)
            label_fuzzy = label_batch[fuzzy_knowledge_indices].view(-1)
            
            for idx, azimuth_temp in enumerate(azimuth_fuzzy):
                
                #    判定候选目标范围，并筛选出符合条件的先验目标信息
                target_candidate_kd, target_candidate_fd, target_candidate_label = target_candidate_searching(train_kd, train_fd, train_azimuth, train_label, azimuth_temp)
                #    从候选目标中找出最符合测试目标分布的先验知识
                prior_knowledge, KLD_sort, indices = prior_knowledge_searching(target_candidate_kd, target_candidate_fd, target_candidate_label, features_fuzzy, idx, vote=True)
                #    模糊性知识融合
                namda = 1
                knowledge_fusion = namda*fuzzy_knowledge[idx] + (1-namda)*prior_knowledge
                fuzzy_knowledge_fusion.append(knowledge_fusion.view(-1, 10))
                
                #   调试start---
                fuzzy_target_candidate_kd.append(target_candidate_kd.data.cpu().numpy())
                fuzzy_target_candidate_fd.append(target_candidate_fd.data.cpu().numpy())
                fuzzy_target_candidate_label.append(target_candidate_label.data.cpu().numpy())
                fuzzy_prior_knowledge.append(prior_knowledge.data.cpu().numpy())
                fuzzy_KLD_sort.append(KLD_sort.data.cpu().numpy())
                fuzzy_indices.append(indices.data.cpu().numpy())
                #   调试end---
            
            fuzzy_knowledge_label.append(label_fuzzy)
            fuzzy_knowledge_azimuth.append(azimuth_fuzzy)
            fuzzy_features.append(features_fuzzy)
            fuzzy_probablity.append(fuzzy_knowledge)

    correct = 0
    #     explicit knowledge recognition
    explicit_knowledge_fusion = torch.cat(explicit_knowledge_fusion, dim=0)
    explicit_knowledge_azimuth = torch.cat(explicit_knowledge_azimuth, dim=0)
    explicit_knowledge_label = torch.cat(explicit_knowledge_label, dim=0)
    _, predicted = torch.max(explicit_knowledge_fusion, 1)
    correct1 = (predicted == explicit_knowledge_label).sum().item()
    Accuracy = correct1 / explicit_knowledge_label.size(0) * 100
    correct += correct1
    
    print('EKR-correct:{}, total:{}, accuracy:{}'.format(correct1, explicit_knowledge_label.size(0), Accuracy))
    # scipy.io.savemat('./results/explicit_knowledge_test2.mat', {'ek':explicit_knowledge_fusion, 'ek_azimuth':explicit_knowledge_azimuth,
                                                               # 'ek_label':explicit_knowledge_label, 'ek_predicted':predicted})
    
    #    general knowledge recognition
    general_knowledge_fusion = torch.cat(general_knowledge_fusion, dim=0)
    general_probability = torch.cat(general_probability, dim=0)
    general_features = torch.cat(general_features, dim=0)
    general_knowledge_azimuth = torch.cat(general_knowledge_azimuth, dim=0)
    general_knowledge_label = torch.cat(general_knowledge_label, dim=0)
    _, predicted = torch.max(general_knowledge_fusion, 1)
    correct2 = (predicted == general_knowledge_label).sum().item()
    Accuracy = correct2 / general_knowledge_label.size(0) * 100
    correct += correct2
    
    # general_target_candidate_kd = torch.cat(general_target_candidate_kd, dim=0)
    # general_target_candidate_fd = torch.cat(general_target_candidate_fd, dim=0)
    # general_target_candidate_label = torch.cat(general_target_candidate_label, dim=0)
    # general_prior_knowledge = torch.cat(general_prior_knowledge, dim=0)
    # general_KLD_sort = torch.cat(general_KLD_sort, dim=0)
    # general_indices = torch.cat(general_indices, dim=0)

    #   ----调试-------
    
    general_target_candidate_kd = np.asarray(general_target_candidate_kd)
    general_target_candidate_fd = np.asarray(general_target_candidate_fd)
    general_target_candidate_label = np.asarray(general_target_candidate_label)
    general_prior_knowledge = np.asarray(general_prior_knowledge)
    general_KLD_sort = np.asarray(general_KLD_sort)
    general_indices = np.asarray(general_indices)
    
    print('GKR-correct:{}, total:{}, accuracy:{}'.format(correct2, general_knowledge_label.size(0), Accuracy))
    # scipy.io.savemat('./results/general_knowledge_test2.mat', {'gk':general_knowledge_fusion, 'gk_azimuth':general_knowledge_azimuth,
                                                               # 'gk_label':general_knowledge_label, 'gk_predicted':predicted})
    general_knowledge_fusion = general_knowledge_fusion.data.cpu().numpy()
    general_knowledge_azimuth = general_knowledge_azimuth.data.cpu().numpy()
    general_knowledge_label = general_knowledge_label.data.cpu().numpy()
    predicted = predicted.data.cpu().numpy()
    general_features = general_features.data.cpu().numpy()
    general_probability = general_probability.data.cpu().numpy()

    scipy.io.savemat('./results/general_knowledge_debug.mat', {'gk_fusion':general_knowledge_fusion, 'gk_probablity':general_probability, 'gk_features':general_features, 
                                                               'gk_azimuth':general_knowledge_azimuth, 'gk_label':general_knowledge_label, 'gk_predicted':predicted,
                                                               'gpk':general_prior_knowledge, 'gtc_kd':general_target_candidate_kd,
                                                               'gtc_fd':general_target_candidate_fd, 'gtc_label':general_target_candidate_label,
                                                               'kld_sort':general_KLD_sort, 'indices':general_indices})
    
    
    #    fuzzy knowledge recognition
    fuzzy_knowledge_fusion = torch.cat(fuzzy_knowledge_fusion, dim=0)
    fuzzy_probablity = torch.cat(fuzzy_probablity, dim=0)
    fuzzy_features = torch.cat(fuzzy_features, dim=0)
    fuzzy_knowledge_azimuth = torch.cat(fuzzy_knowledge_azimuth, dim=0)
    fuzzy_knowledge_label = torch.cat(fuzzy_knowledge_label)
    _, predicted = torch.max(fuzzy_knowledge_fusion, 1)
    correct3 = (predicted == fuzzy_knowledge_label).sum().item()
    Accuracy = correct3 / fuzzy_knowledge_label.size(0) * 100
    
    #----调试----
    fuzzy_target_candidate_kd = np.asarray(fuzzy_target_candidate_kd)
    fuzzy_target_candidate_fd = np.asarray(fuzzy_target_candidate_fd)
    fuzzy_target_candidate_label = np.asarray(fuzzy_target_candidate_label)
    fuzzy_prior_knowledge = np.asarray(fuzzy_prior_knowledge)
    fuzzy_KLD_sort = np.asarray(fuzzy_KLD_sort)
    fuzzy_indices = np.asarray(fuzzy_indices)

    print('FKR-correct:{}, total:{}, accuracy:{}'.format(correct3, fuzzy_knowledge_label.size(0), Accuracy))
    # scipy.io.savemat('./results/fuzzy_knowledge_test2.mat', {'fk':fuzzy_knowledge_fusion, 'fk_azimuth':fuzzy_knowledge_azimuth,
                                                             # 'fk_label':fuzzy_knowledge_label, 'fk_predicted':predicted})
    
    fuzzy_knowledge_fusion = fuzzy_knowledge_fusion.data.cpu().numpy()
    fuzzy_knowledge_azimuth = fuzzy_knowledge_azimuth.data.cpu().numpy()
    fuzzy_knowledge_label = fuzzy_knowledge_label.data.cpu().numpy()
    predicted = predicted.data.cpu().numpy()
    fuzzy_features = fuzzy_features.data.cpu().numpy()
    fuzzy_probablity = fuzzy_probablity.data.cpu().numpy()
                                                               
    scipy.io.savemat('./results/fuzzy_knowledge.mat', {'fk_fusion':fuzzy_knowledge_fusion, 'fk_probablity':fuzzy_probablity, 'fk_features':fuzzy_features,
                                                           'fk_azimuth':fuzzy_knowledge_azimuth, 'fk_label':fuzzy_knowledge_label, 'fk_predicted':predicted,
                                                           'fpk':fuzzy_prior_knowledge, 'ftc_kd':fuzzy_target_candidate_kd,
                                                           'ftc_fd':fuzzy_target_candidate_fd, 'ftc_label':fuzzy_target_candidate_label,
                                                           'kld_sort':fuzzy_KLD_sort, 'indices':fuzzy_indices})
    
    
    #  总体测试
    correct += correct3
    Accuracy = correct / len(testlabel) * 100
    
    print('Total-correct:{}, total:{}, accuracy:{}'.format(correct, len(testlabel), Accuracy))
    
    







