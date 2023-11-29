
""" 
       training on Knowledge Dissemination Network---KDNet
"""

# import
from KDNets import *
from tools import Class_AzimuthLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#---------------------__Main__-----------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')
    
#  hyperparameter setting        
batch_size = 32
num_classes = 10
num_epochs = 500
learning_rate = 1e-4
weight_decay = 1e-4
seed = 101

#   seeds initialization
seeds_init(seed)

#----------------DataLoader-----------------
train_dataset = scipy.io.loadmat('./dataset/data_train_128.mat')
test_dataset  = scipy.io.loadmat('./dataset/data_test_128.mat')

traindata_am = train_dataset['data_am']
traindata_azimuth = np.int16(train_dataset['azimuth'])
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量

testdata_am = test_dataset['data_am']
testdata_azimuth = np.int16(test_dataset['azimuth'])
testlabel = test_dataset['label'].squeeze()

train_dataset = MyDataset(img=traindata_am, azimuth=traindata_azimuth, label=trainlabel, transform=transforms.ToTensor())
test_dataset  = MyDataset(img=testdata_am, azimuth=testdata_azimuth, label=testlabel, transform=transforms.ToTensor())
print('train data size: {}'.format(train_dataset.img.shape[0]))
print('test data size: {}'.format(test_dataset.img.shape[0]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#---------------------model preparation------------------
model = KDNet(num_classes).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = nn.NLLLoss()
# criterion = Class_AzimuthLoss()
criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, \
             # verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.8)
milestones = [100, 150, 250]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.3)
#-------------------------training-----------------------
print('--------------training...-----------------')
train_loss = []
train_acc = []
# train_classloss = []
# train_angleloss = []
val_loss = []
val_acc = []
# val_classloss = []
# val_angleloss = []
total_step = len(trainlabel) // batch_size
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for batch_idx, (image, azimuth, label) in enumerate(train_loader):
        
        images   = image.to(device)
        azimuth = azimuth.to(device)        
        label   = label.to(device)
        
        optimizer.zero_grad()
        output, features, attn = model(images, azimuth)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx+1) % 20 == 0:
            print ('LR={}, Epoch [{}/{}], Step [{}/{}], Step Loss: {:.8f},  Total Loss: {:.8f}' 
                   .format(optimizer.param_groups[0]['lr'], epoch+1, num_epochs, batch_idx+1, total_step, loss.item(), total_loss))                     
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    scheduler.step()     ## 自适应动态调整学习率
    print('---------------------------training-----------------------------')    
    print('correct number : {}, train data number : {}, Accuracy : {:.4f}, train loss: {:.6f}'.format(correct, total, 100 * correct / total, total_loss))   
    train_acc.append(correct/total)
    train_loss.append(total_loss)
    #  save model
    # if correct/total == 1:
    # if (epoch+1) % 10 == 0:
        # acc = ('%.4f'%(correct/total))
        # savepath = './models/knowledge_models/fullmodel_'+str(epoch+1)+'Ep_'+acc+'Acc.pth'
        # torch.save(model,savepath)    
    
    #----------------Validation----------------
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        temp_loss = 0
        labels = []
        label_pre = []
        for image, azimuth, label in test_loader:
            image   = image.to(device)
            azimuth = azimuth.to(device)			
            label   = label.to(device)

            output, features, attn = model(image, azimuth)
            loss = criterion(output, label)
            temp_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            labels.append(label)
            label_pre.append(predicted)
            # label.extend(target.data.cpu().numpy())      # data form GPU to CPU
            # label_pre.extend(predicted.data.cpu().numpy())
        print('---------------------------validation---------------------------')  
        print('correct number : {}, test data number : {}, Accuracy : {:.4f}, test loss: {:.6f}'.format(correct, total, 100 * correct / total, temp_loss))
        print('----------------------------------------------------------------')
        # print('Training Loss: {},  Valdation Loss: {}'.format(total_loss, temp_loss))
        # print('class loss:{}, azimuth loss:{}, total loss:{}'.format(classloss, angleloss, temp_loss))
        # print('concat weights:{}'.format(model.Concat.namda.data))
        print('[x_up, x_down, y_up, y_down]={}'.format(model.KDM.alpha.data))
        print('----------------------------------------------------------------\n')
        print('****************************************************************')
        val_loss.append(temp_loss)
        val_acc.append(correct/total)

    #  save model
    if (correct/total) > 0.997:
        acc = ('%.4f'%(correct/total))
        savepath = './models/fullmodel_'+str(epoch+1)+'Ep_'+acc+'Acc.pth'
        torch.save(model,savepath)
val_acc_max, idx = torch.max(torch.Tensor(val_acc), -1)
val_loss1 = torch.Tensor(val_loss)[idx]
print('KDNet: val_acc: {}, val_loss: {}, idx: {}'.format(val_acc_max, val_loss1, idx+1))           
#-----------trian loss curve--------------
plt.figure#(figsize=(10,5.625))
plt.title('train and val loss curves on KDNets', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.tick_params(labelsize=10)   #调整坐标轴刻度的字体大小
plt.legend(fontsize=10)       #参数调整train-loss与val-loss字体的大小
plt.savefig("./results/fig1.jpg")
plt.show()

plt.figure#(figsize=(10,5.625))
plt.title('train and val acc curves on KDNets', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Acc', fontsize=15)
plt.plot(train_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.tick_params(labelsize=10)   #调整坐标轴刻度的字体大小
plt.legend(fontsize=10)       #参数调整train-loss与val-loss字体的大小
plt.savefig("./results/fig2.jpg")
plt.show()
#---------------save model----------------
# torch.save(model,'./models/fullmodel_100Ep_1e-3lr.pth')
    
    


















 