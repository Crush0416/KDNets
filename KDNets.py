"""
       This is a project for SAR image recognition with Knowledge Dissemination Networks-KDNets
"""

# import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import os

# import time

# ---------------------Functions && classes----------------------------

def seeds_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  ##  CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def attributed_scatter_center(target):
    c, h, w = target.shape
    B = torch.tensor(5.91e8)    # bandwidth 0.591GHz
    c = torch.tensor(3e8)       # light speed 3*10^8
    fc = torch.tensor(9.6e9)    # center frequency 9.6GHz
    phi = torch.tensor(1.76*2)  # aspect angle
    x = torch.tensor([i for i in range(h)]).reshape(h,1).cuda()
    y = torch.tensor([i for i in range(w)]).reshape(1,w).cuda()
    
    sc_index = torch.where(target == torch.max(target))   # 获取最大散射点索引

    sc_out = torch.zeros(h,w).cuda()
    if len(sc_index[0]) > 1:
        for i in range(0, len(sc_index[0])):
            sc_temp = target[sc_index[0][i], sc_index[1][i], sc_index[2][i]] * (torch.sinc((2 * B / c) * (x - sc_index[1][i])) * torch.sinc(
                (2 * 2 * fc * torch.sin(phi / 2 / 180) / c) * (y - sc_index[2][i])))  # 计算ASC
            sc_out += sc_temp
    else:
        sc_out = target[sc_index] * (torch.sinc((2 * B / c) * (x - sc_index[1])) * torch.sinc((2 * 2 * fc * torch.sin(phi / 2 / 180) / c) * (y - sc_index[2])))  # 计算ASC
    
    return sc_out        
    
class KnowledgeDissemination(nn.Module):
    def __init__(self, target_size, targetSave_size, targetCut_size):
        super(KnowledgeDissemination, self).__init__()
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)

        self.target_h, self.target_w = target_size  # 目标默认尺寸：(40,20)
        self.targetSave_size = targetSave_size  # 目标保存尺寸：(48*48)
        self.targetCut_size = targetCut_size  # 均衡处理后目标裁剪尺寸：(64*64)
    
    def TargetCut(self, images):
        b, c, h, w = images.shape
        x_up = int(h / 2 + self.targetCut_size / 2)
        x_down = int(h / 2 - self.targetCut_size / 2)
        y_up = int(w / 2 + self.targetCut_size / 2)
        y_down = int(w / 2 - self.targetCut_size / 2)
        full_ones = torch.ones(b, c, h, w).cuda()
        imageEq = torch.where(images * 2 >= 1, full_ones, images * 2)  # 2 equilibrium
        output = imageEq[:, :, x_down:x_up, y_down:y_up]  # corpping, 64*64
        return output
    
    def mask_making(self, azimuth, indices, size):
        # x, y = indices
        x, y = 64, 64   # image center
        mask = torch.zeros(size).cuda()  # genreate an all-zeros matrix of the same size as input image
        self.x_up = torch.round(x + self.alpha[0] * self.target_h / 2).int()  # 长默认为40，一半为20；宽默认为20，一半为10，后续再调整
        self.x_down = torch.round(x - self.alpha[1] * self.target_h / 2).int()
        self.y_up = torch.round(y + self.alpha[2] * self.target_w / 2).int()
        self.y_down = torch.round(y - self.alpha[3] * self.target_w / 2).int()
        mask[:, self.x_down:self.x_up, self.y_down:self.y_up] = 1  # set mask=1
        mask = TF.rotate(img=mask, angle=-azimuth.item())  # rotate the mask
        return mask
    
    def target_separating(self, images, azimuth):
        b, c, h, w = images.shape
        target = []
        target_scene = []
        for i, img in enumerate(images):
            scatter_center = torch.where(img.squeeze() == torch.max(img))  # 获取最大散射点索引
            mask = self.mask_making(azimuth[i], scatter_center, (c, h, w))
            target_temp = img * mask
            
            # azimuth_temp = azimuth[i] / 360 * mask     #方位角归一化
            # target_temp = target_temp + azimuth_temp   #加入方位角
            
            x_up = int(h / 2 + self.targetSave_size / 2)
            x_down = int(h / 2 - self.targetSave_size / 2)
            y_up = int(h / 2 + self.targetSave_size / 2)
            y_down = int(h / 2 - self.targetSave_size / 2)
            target_temp = target_temp[:, x_down:x_up, y_down:y_up]  # 将目标区域裁剪为48*48
            target.append(target_temp)
            # target = torch.cat((target, target_temp))
            mask_scene = torch.where(mask == 0, 1, 0)  # 对mask取反操作，1变为0.0变为1.作为背景mask
            scene = img * mask_scene
            # maxvalue = torch.max(scene)
            # scene = maxvalue - scene
            # scene = torch.where(scene.squeeze() == maxvalue, torch.zeros(h,w).cuda(), scene.squeeze())
            target_scene.append(scene)
            # target_scene = torch.cat((target_scene, scene))
        target = torch.cat(target, dim=0).reshape(b,c,self.targetSave_size,self.targetSave_size)
        target_scene = torch.cat(target_scene, dim=0).view_as(images)
            
        return target, target_scene
    
    def target_shadow(self, scene):
        b, c, h, w = scene.shape
        target_shadow = []
        kernel_size = 2

        for img in scene:
            img = img.squeeze()
            shadow = torch.zeros(h,w)
            mean = torch.mean(img)
            for i in range(h-kernel_size):
                for j in range(w-kernel_size):
                    aa = torch.mean(img[i:i+kernel_size, j:j+kernel_size]) 
                    if torch.mean(img[i:i+kernel_size, j:j+kernel_size]) < mean-0.03:
                        shadow[i:i+kernel_size, j:j+kernel_size] = img[i:i+kernel_size,j:j+kernel_size]
            target_shadow.append(shadow)        
        target_shadow = torch.cat(target_shadow, dim=0).view_as(scene)
        #  画图保存
        # target_shadow1 = target_shadow.data.cpu().numpy()
        # for ii in range(b):
        #     plt.imshow(target_shadow1[ii, :, :, :].squeeze())
        #     plt.savefig('./results/target_shadow'+str(ii)+'.PNG', bbox_inches='tight', dpi=500)
        
        return target_shadow
        
    def scatter_center_extraction(self, targets):
        b, c, h, w = targets.shape
        tol = 0.1  # CLEAN策略停止条件
        target_asc = []
        for img in targets:
            scatter_center = attributed_scatter_center(img)
            scatter_center_residual = img - scatter_center
            energy_ratio = torch.sum(torch.pow(scatter_center_residual,2)) / torch.sum(torch.pow(img,2))    # 计算能量比
            while energy_ratio >= tol:
                scatter_center_temp = attributed_scatter_center(scatter_center_residual)
                scatter_center += scatter_center_temp
                scatter_center_residual = img - scatter_center
                energy_ratio = torch.sum(torch.pow(scatter_center_residual,2)) / torch.sum(torch.pow(img,2))    # 计算能量比
            target_asc.append(scatter_center)    
        target_asc = torch.cat(target_asc, dim=0).view_as(targets)
        
        return target_asc                       
		
    def forward(self, image, azimuth):
        b, c, h, w = image.shape
        target, target_scene = self.target_separating(image, azimuth)
        target_cut = self.TargetCut(image)
        target_raw = image
        # target_asc = self.scatter_center_extraction(target)
        
        # target_shadow = self.target_shadow(target_scene)
        '''
        # 画图保存
        target = target.data.cpu().numpy()
        target_scene = target_scene.data.cpu().numpy()
        target_cut = target_cut.data.cpu().numpy()
        target_asc = target_asc.data.cpu().numpy()
        
        for ii in range(b):
            plt.imshow(target[ii, :, :, :].squeeze())
            plt.savefig('./results/target'+str(ii)+'.PNG', bbox_inches='tight', dpi=500)
            # plt.show()
            plt.imshow(target_scene[ii, :, :, :].squeeze())
            plt.savefig('./results/target_scene' + str(ii) + '.PNG', bbox_inches='tight', dpi=500)

            plt.imshow(target_cut[ii, :, :, :].squeeze())
            plt.savefig('./results/target_cut' + str(ii) + '.PNG', bbox_inches='tight', dpi=500)
            
            plt.imshow(target_asc[ii, :, :, :].squeeze())
            plt.savefig('./results/target_asc'+str(ii)+'.PNG', bbox_inches='tight', dpi=500)
            # plt.show()
        '''
        return target, target_cut, target_scene, target_raw

class Local_Information_Stream(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, padding=1, interp_size=(8,8)):
        super(Local_Information_Stream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))              
            
        self.interp = nn.Upsample(size=interp_size, mode='bilinear', align_corners=True)
    def forward(self, target):
        
        residule = self.downsample1(target)
        out = self.layer1(target)       
        out += residule
        out = F.relu(out)
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample3(out)
        out = self.layer3(out)
        out += residule
        out = F.relu(out)
        
        out = self.interp(out)
        return out
'''
#       target asc stream
class Target_ASC_Stream(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, padding=1, interp_size=(8,8)):
        super(Target_ASC_Stream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))   
        self.interp = nn.Upsample(size=interp_size, mode='bilinear', align_corners=True)

    def forward(self, target_asc):
        
        residule = self.downsample1(target_asc)
        out = self.layer1(target_asc)       
        out += residule
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        
        residule = self.downsample3(out)
        out = self.layer3(out)
        out += residule
        
        out = self.interp(out)
        return out   
'''

class Visual_Information_Stream(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=5, padding=2):
        super(Visual_Information_Stream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),)
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))  

    def forward(self, target_equilibrium):
        
        residule = self.downsample1(target_equilibrium)
        out = self.layer1(target_equilibrium)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample3(out)
        out = self.layer3(out)
        out += residule
        out = F.relu(out)

        return out     

class Shadow_Information_Stream(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=7, padding=3):
        super(Shadow_Information_Stream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=4),
            nn.BatchNorm2d(out_channels))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=4),
            nn.BatchNorm2d(out_channels))

    def forward(self, target_scene):
        
        residule = self.downsample1(target_scene)
        out = self.layer1(target_scene)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        out = F.relu(out)

        return out          

#     
class Global_Information_Stream(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=7, padding=3):
        super(Global_Information_Stream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            )
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.downsample4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))

    def forward(self, target_raw):
        
        residule = self.downsample1(target_raw)
        out = self.layer1(target_raw)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample3(out)
        out = self.layer3(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample4(out)
        out = self.layer4(out)
        out += residule
        out = F.relu(out)

        return out                                       
        
class Concat_Fusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, kernel_size=5, padding=2):
        super(Concat_Fusion, self).__init__()
        self.namda = nn.Parameter(torch.ones(4), requires_grad=True)
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            )
        
    def forward(self, K1, K2, K3, K4):
        out = torch.cat([self.namda[0]*K1, self.namda[1]*K2, self.namda[2]*K3, self.namda[3]*K4], dim=1)
        # out = torch.cat([K1, K2, K3, K4], dim=1)
        out = self.layer1(out)
        return out        

#--------数据流转换--------
class stream_transpose(nn.Module):
    def __init__(self, s_num=4):
        super(stream_transpose, self).__init__()
        self.s = s_num
        
    def forward(self, x1, x2, x3, x4):
        b, c, h, w = x1.size()
        s = self.s   # 数据流数量
        x = torch.cat([x1, x2, x3, x4], dim=1).view(b, s, c, h*w)   # 转化为[b, s, c, embding_dim], embding_dim=h*w
        
        return x, s
        
#-----------Stream Self-Attention-------------
class SSA(nn.Module):
    def __init__(self, embding_dim, k_dim, streams):
        super(SSA, self).__init__()
        self.k_dim=k_dim
        self.s = streams #通道数
        
        self.query = nn.Linear(embding_dim, k_dim)    # nn.Linear()支持高维线性运算。维度之于最后一维有关
        self.key = nn.Linear(embding_dim, k_dim)
        self.value = nn.Linear(embding_dim, k_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.LN = nn.LayerNorm([k_dim])
        self.BN = nn.BatchNorm2d(self.s)
        
    def forward(self, x):
        b, s, c, embding_dim = x.size()   # embding_dim=h*w
        
        q = self.query(x).view(b,s,-1)   #输出[b,s,c*k_dim]
        k = self.key(x).view(b,s,-1) 
        v = self.value(x).view(b,s,-1) 

        scores = torch.matmul(q, k.permute(0,2,1)) / torch.sqrt(torch.tensor(self.k_dim))
        attention = self.softmax(scores)
        
        content = torch.matmul(attention, v).view(b,s,c,self.k_dim)   # 矩阵乘
        output = self.LN(content+x)
        
        return output, attention

#-----------Multi Heads Stream Self-Attention-------------
class MHSSA(nn.Module):
    def __init__(self, embding_dim, k_dim, heads, streams):
        super(MHSSA, self).__init__()
        self.heads = heads
        self.k_dim = k_dim
        self.s = streams  
        
        self.query = nn.Linear(embding_dim, k_dim*heads)    # nn.Linear()支持高维线性运算。维度只与最后一维有关
        self.key = nn.Linear(embding_dim, k_dim*heads)
        self.value = nn.Linear(embding_dim, k_dim*heads)
        
        self.softmax = nn.Softmax(dim=-1)
        self.weights = nn.Linear(k_dim*heads, k_dim)
        
        self.LN = nn.LayerNorm([k_dim])
        self.BN = nn.BatchNorm2d(self.s)
        
    def forward(self, x):
        b, s, c, embding_dim = x.size()
        heads = self.heads
        k_dim = self.k_dim
        
        q = self.query(x).view(b, s, c, heads, -1).transpose(1,3)   #输出[b,heads,c,s,k_dim]，PS:如果当前MHSSA效果不佳，可以考虑转化为[b,heads,c,s,k_dim]去计算
        k = self.key(x).view(b, s, c, heads, -1).transpose(1,3) 
        v = self.value(x).view(b, s, c, heads, -1).transpose(1,3) 

        scores = torch.matmul(q, k.permute(0,1,2,4,3)) / torch.sqrt(torch.tensor(self.k_dim*self.heads))  #输出[b,heads,c,s,s]
        attention = self.softmax(scores)
        
        content = torch.matmul(attention, v)   # 矩阵乘   [b,heads,c,s,k_dim]
        content = content.permute(0,3,2,1,4).reshape(b, s, c, heads*k_dim)  #[b,s,c,heads*k_dim]
        
        content = self.weights(content).view(b,s,c,k_dim)  # [b,s,c,k_dim]
        contene = self.LN(content+x)
        
        return content, attention

class FeedForwardLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        
        output = self.fc_layer(x)
        
        return output

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels, out_channels, kernel_size=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        
        output = self.conv_layer(x)
        
        return output
        
# --------------------- Architecture-----------------------
class KDNet(nn.Module):
    ''' this is the backbone network of Multi-Streams Complex Value Networks'''
    
    def __init__(self, num_classes):
        super(KDNet, self).__init__()        
        
        # ---------------konwledge dissemination module-------------#
        self.KDM = KnowledgeDissemination(target_size=(40, 20), targetSave_size=48, targetCut_size=64)
        
        # ---------------Knowledge Stream-多流多尺度知识提取---------------------#
        self.LIS    = Local_Information_Stream(in_channels=1, out_channels=64, kernel_size=3, padding=1, interp_size=(8,8))
        # self.TASCS = Target_ASC_Stream(in_channels=1, out_channels=64, kernel_size=3, padding=1, interp_size=(8,8))
        self.VIS   = Visual_Information_Stream(in_channels=1, out_channels=64, kernel_size=5, padding=2)      
        self.SIS   = Shadow_Information_Stream(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        self.GIS   = Global_Information_Stream(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        
        # ----------------Knowledge Fusion--------------------------#
        # self.Concat = Concat_Fusion(in_channels=256, out_channels=128, kernel_size=5, padding=2)

        # ----------------stream_transpose----------------
        self.ST = stream_transpose(s_num=4)
        
        #-----------------Stream Self Attention------------------------#
        self.SSA = SSA(embding_dim=64, k_dim=64, streams=4)
        # self.MHSSA = MHSSA(embding_dim=64, k_dim=64, heads=8, streams=4)
        
        #-----------------Conv Layer--------------------------#
        self.conv_layer = ConvLayer(in_channels=256, out_channels=128, dropout=0.5)          
        
        # ----------------Full Connection Layers----------------#
        self.FFL = FeedForwardLayer(in_features=128, out_features=512, dropout=0.7)
        
        self.classifier = nn.Linear(128, num_classes)
        self.classifier_relu = nn.ReLU()
    
    def forward(self, images, azimuth):
        b, c, h, w = images.shape      # [32,1,128,128]
        
        LIS, VIS, SIS, GIS = self.KDM(images, azimuth)   # 知识分发模块：
                                                                                   # LIS-48*48, 
                                                                                   # VIS-64*64, 
                                                                                   # SIS-128*128
                                                                                   # GIS-128*128
        out1 = self.LIS(LIS)
        out2 = self.VIS(VIS)
        out3 = self.SIS(SIS)
        out4 = self.GIS(GIS)
        b, c, h, w = out1.size()                   # [32,64,8,8]
        
        #------------方位角中间级融合--------------
        out1 = out1 + (azimuth / 360).view(b,1,1,1)
        
        out, s = self.ST(out1, out2, out3, out4)      # [b,s,c,h*w]
        #--------加入方位信息------
        # out = out + (azimuth/360).view(b,1,1,1)
        # out = out * (azimuth/360).view(b,1,1,1)
        out, attn = self.SSA(out)
        # out, attn = self.MHSSA(out)                  # [b,s,c,h*w], [32,4,64,64]
        
        out = out.view(b, s*c, h, w)
        features = self.conv_layer(out).squeeze()
        # out = out.flatten(1,-1)                    # [b,s*c*h*w]
        # out = self.FFL(out)
        
        #-------------方位角决策级融合-----------------
        # features = features + (azimuth / 360)
        
        out = self.classifier(features)     #[batch_size, num_classes]
        return out, features, attn


# ---------------------MyDataset-----------------------
class MyDataset(Dataset):
    def __init__(self, img, azimuth, label, transform=None):
        super(MyDataset, self).__init__()
        self.img = torch.from_numpy(img).float()
        self.azimuth = torch.from_numpy(azimuth).float()
        self.label = torch.from_numpy(label).long()
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.img[index]
        azimuth = self.azimuth[index]
        label = self.label[index]
        return img, azimuth, label
    
    def __len__(self):
        return self.img.shape[0]
