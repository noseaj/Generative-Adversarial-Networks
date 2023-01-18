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
            load_file =pickle.load(f)
        return load_file
        
    def check_img(self, name, leak_center_data):
        for index, (key, elem) in enumerate(leak_center_data.items()):
            if key in name:
                return elem

    def extraction_x_y(self, centered_leak):
        i =random.randint(0,len(centered_leak))
        return centered_leak[i]
    
    def generate(self, leak, centered_leak):
        img = cv2.imread(self.img)
        leak_img = cv2.imread(leak)
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
    
leak_center_data = {'L_01-1_T': 'L_01_1_Top.pkl', 'L_01-1_R': 'L_01_1_Rotation.pkl',
                    'L_01-2_T': 'L_01_2_Top.pkl', 'L_01-2_R': 'L_01_2_Rotation.pkl',
                    'L_01-3_T': 'L_01_3_Top.pkl', 'L_01-3_R': 'L_01_3_Right.pkl',
                    'L_01-4_T': 'L_01_4_Top.pkl', 'L_01-4_R': 'L_01_4_Right.pkl',
                    'L_01-5_T': 'L_01_5_Top.pkl', 'L_01-5_R': 'L_01_5_Right.pkl',
                    'L_01-6_T': 'L_01_6_Top.pkl', 'L_01-6_R': 'L_01_6_Right.pkl'}

def main(Name):
    img = "./public/images/test.jpg" 
    leak_img = Generate_Leak_Img(img)

    # 정상 이미지 타입 확인
    img_type = leak_img.check_img(Name, leak_center_data)
    
    # 해당 이미지 타입에서 발생했던 리크 위치
    typical_leak = leak_img.open_list(img_type)
    
    # 생성할 위치 좌표
    centered_leak = leak_img.extraction_x_y(typical_leak)

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
    
    generated_leak_img = leak_img.generate(leak, centered_leak)
    cv2.imwrite("./public/images/result.jpg", generated_leak_img) 

if __name__ == '__main__':
    main(sys.argv[1])
