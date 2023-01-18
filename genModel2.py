import sys
import numpy as np
import cv2
import pickle
import matplotlib.pylab as plt
import random
import copy

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    
class Generate_Leak_Img:
    def __init__(self, img):
        self.img = img
            
    def open_list(self, file_name):
        with open(file_name,'rb') as f:
            load_file = pickle.load(f)
        return load_file
        
    def check_img(self, leak_center_data):
        for index, (key, elem) in enumerate(leak_center_data.items()):
            if key in self.img:
                return elem

    def extraction_x_y(self, centered_leak):
        i =random.randint(0,len(centered_leak))
        return centered_leak[i]
    
    def generate(self, leak, centered_leak, resize_x, resize_y):
        img = cv2.imread(self.img)
        leak_img = cv2.imread(leak)
        leak_img = cv2.resize(leak_img,(int(resize_x),int(resize_y)))
            
        # leak 생성할 좌표
        width = centered_leak['x']
        height = centered_leak['y']
        center = (int(width), int(height))
        
        # 마스크 생성, 합성할 이미지 전체 영역을 255로 세팅
        pix_value = int(np.mean(img[int(height),int(width)]))
        mask = np.full_like(leak_img, pix_value)
        
        # seamlessClone으로 합성
        mixed = cv2.seamlessClone(leak_img, img, mask, center, cv2.MIXED_CLONE)
        return mixed
    
def main(X, Y, resize_x, resize_y):
    print(X, Y)
    print(resize_x, resize_y)
    img = "./public/images/test.jpg" 
    leak_img = Generate_Leak_Img(img)
    centered_leak = {'x': X, 'y': Y}

    # 1. Generator Class 객체 생성
    device = torch.device('cpu')
    ngpu = 0
    Generator_Leak = Generator(ngpu).to(device)
    
    # 2. 모델 사용
    weights = torch.load('./dcgan_293.pt',map_location=device)
    Generator_Leak.load_state_dict(weights)
    
    # 3. 리크 생성
    noise = torch.randn(1, 100, 1, 1, device=device)  
    leak = Generator_Leak(noise)
    save_image(leak, './public/images/gen_leak.jpg')
    leak = './public/images/gen_leak.jpg'
    
    generated_leak_img = leak_img.generate(leak, centered_leak, resize_x, resize_y)
    cv2.imwrite("./public/images/result.jpg", generated_leak_img) 

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) 
