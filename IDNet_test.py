import numpy as np
from KDNets import *
from tools import Class_AzimuthLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os   

#----------Functions-----------
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
path = './models/models_relu/class10/SSA/fullmodel_198Ep_0.9963Acc.pth'
batch_size = 100
num_classes = 10

#---------------DataLoader-------------------
train_dataset = scipy.io.loadmat('../../Datasets/Azimuth_MSTAR/data_SOC/class10/train/data_train_128.mat')
test_dataset  = scipy.io.loadmat('../../Datasets/Azimuth_MSTAR/data_SOC/class10/test/data_test_128.mat')

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

#-------------IDNet test--------------    
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0  
    label = []
    label_pre = []
    attention = []
    
    for image_batch, azimuth_batch, label_batch in test_loader:
        image_batch   = image_batch.to(device)
        azimuth_batch = azimuth_batch.to(device)
        label_batch   = label_batch.to(device)

        output, features, attn = model(image_batch, azimuth_batch)
        loss = criterion(output, label_batch)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1) 
        total += image_batch.size(0)
        correct += (predicted == label_batch).sum().item()           
        label.extend(label_batch.data.cpu().numpy())
        label_pre.extend(predicted.data.cpu().numpy())
        attention.append(attn)
    attention = torch.cat(attention, dim=0)
    print('correct number : {}, test data number : {}, test Accuracy: {}, test loss: {:.4f}'.format(correct, total, 100*correct/total, total_loss))    
    # label = np.asarray(label)
    # label_pre = np.asarray(label_pre)
    matrix = confusion_matrix(label,label_pre)
    print('############ confusion matrix ########### \n', matrix)
    print('------------Attentions--------------\n', attention[100,:,:])
    # scipy.io.savemat('./results/confusion_matrix_class3.mat',{'confusion_matrix':matrix,'label':label, 'label_predict':label_pre})        
        
     
    
    







