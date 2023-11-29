import torch
import torch.nn as nn

class Class_AzimuthLoss(nn.Module):
    def __init__(self):
        super(Class_AzimuthLoss, self).__init__()
        self.alpha = 1.       
        self.ClassLoss = nn.CrossEntropyLoss(ignore_index=255)
        self.AzimuthLoss = nn.MSELoss()
        
    def forward(self, output, label, azimuth):
        azimuth = azimuth.squeeze()/360    # 归一化
        out_classes, out_azimuth = output[:,:-1], output[:,-1]
        Loss1 = self.ClassLoss(out_classes, label)
        Loss2 = self.alpha*self.AzimuthLoss(out_azimuth, azimuth)
        Loss = self.ClassLoss(out_classes, label) + self.alpha*self.AzimuthLoss(out_azimuth, azimuth)
        return Loss, Loss1, Loss2
        