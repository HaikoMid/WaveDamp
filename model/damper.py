import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import ptwt, pywt
import torch.nn.functional as F
from torchvision.transforms import v2
import random
    

class ResCon(nn.Module):
    def __init__(self, channel_in, kernel_size):
        super(ResCon, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=18, kernel_size=kernel_size, stride=1, padding='same', groups=3)
        self.PReLU1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=kernel_size, stride=1, padding='same', groups=3)
        self.PReLU2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=36, out_channels=channel_in, kernel_size=kernel_size, stride=1, padding='same', groups=3)
        self.PReLU3 = nn.PReLU()
        
    def forward(self, x):
        residual = x
        out = self.PReLU1(self.conv_1(x))
        out = self.PReLU2(self.conv_2(out))
        out = self.PReLU3(self.conv_3(out))
        out = torch.add(out, residual)
        return out


class WaveDamper(nn.Module):
    def __init__(self, wavelet, level, min_severity=1, max_severity=1):
        super(WaveDamper, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.min_severity = min_severity
        self.max_severity = max_severity
        self.conv1 = nn.ModuleList([ResCon(9, 3+2*i) for i in reversed(range(self.level))])

    def forward(self, x):
        fc_features = []
        severity = random.uniform(self.min_severity, self.max_severity)

        # Decompose the image into wavelet coefficients for each level
        x_new = x
        for i in range(self.level):
            LL, (LH, HL, HH) = ptwt.wavedec2(x_new, pywt.Wavelet(self.wavelet), level=1)
            FC = torch.cat((LH, HL, HH), dim=1)
            damped_FC = self.conv1[i](FC)
            fc_features.append(damped_FC)
            x_new = LL  # Update for the next level

        LL = x_new
        HFC = []
    
        for i in reversed(range(self.level)):
            LH, HL, HH = torch.split(fc_features[i], 3, dim=1)
            HFC.append((LH, HL, HH))
            
        rec_image = ptwt.waverec2([torch.zeros_like(LL)] + HFC, pywt.Wavelet(self.wavelet))

        return x - torch.mul(rec_image, severity)


def load_damper(opt):
    damper = WaveDamper(wavelet=opt.wavelet, level=opt.level, min_severity=opt.severity)
    checkpoint = torch.load(f'pretrained/{opt.damper_name}')['state_dict']
    if 'damper' in opt.aug:
        checkpoint_keys = list(checkpoint.keys())
        for key in checkpoint_keys:
            checkpoint[key.replace('model.backbone.', '')] = checkpoint[key]
            del checkpoint[key]
        damper.load_state_dict(checkpoint, strict=True)
        damper.eval()
    return damper


if __name__ == '__main__':
    net=WaveDamper(wavelet='bior2.2', level=4).cuda()
    y = net(Variable(torch.randn(32,3,224,224).cuda()))
    print(y.size())
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')