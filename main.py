from utils.preprocessing import Preprocessing
from models.generator import Generator
import easydict
import torch
from models.discriminator import Discriminator

from utils.dataloader import Dataloader
from models.train import Train
import torch.nn as nn
import torch.optim as optim


opt = easydict.EasyDict({
    'workers' : 2,
    'batchSize' : 183,
    'imageSize' : 64,
    'nc' : 3,
    'nz' : 100,
    'ngf' : 64,
    'ndf' : 64,
    'niter' : 50,
    'lr' : 0.0005,
    'beta1' : 0.5,
    'cuda' : 'store_true',
    'ngpu' : 1,
    'outf' : '.',
    'real_label' : 1,
    'fake_label' : 0
})


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
    
    pre = Preprocessing()
    
    # 전처리 1
    pre.leak_extract()
    
    # 전처리 2
    leak_file_path , leak_file_list = pre.remove_grid_line()
    
    # 전처리 3
    resize_file_path, resize_file_list = pre.resize(leak_file_path , leak_file_list)
    
    # 전처리 4
    modeling_file_path = pre.augmentation(resize_file_path, resize_file_list)
    
    # 데이터 불러오기
    loader = Dataloader(opt)
    trainloader,testloader,dataloader,testset = loader.setData()
    
    # 모델 생성
    # Create the Generator
    netG = Generator(opt).to(device)
    netG.apply(weights_init) # Apply the weights_init function to randomly initialize all weights
    
    # Create the Discriminator
    netD = Discriminator(opt).to(device)
    netD.apply(weights_init) # Apply the weights_init function to randomly initialize all weights
    
    # 모델 훈련
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    train = Train(opt, netD,netG, trainloader, testloader, criterion, optimizerD, optimizerG, testset)
    
    train.Train(device)
    
    
    
if __name__ == '__main__':
    main()