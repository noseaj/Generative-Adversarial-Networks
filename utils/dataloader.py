import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
import os

class Dataloader():
    def __init__(self, opt):
        self.dataroot = 'data\Leak_file_path\Resize\\'
        self.opt = opt
        

    def setData(self):
        # Data Preprocessing. Normalizing the pixels and tensors
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = dset.ImageFolder(root=self.dataroot, transform=transform)
        assert trainset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.opt.batchSize, shuffle=True, num_workers=int(self.opt.workers))
        
        testset = dset.ImageFolder(root=self.dataroot, transform=transform)
        assert testset
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.opt.batchSize, shuffle=False, num_workers=int(self.opt.workers))

        dataloader = torch.utils.data.DataLoader("leak", batch_size=self.opt.batchSize, shuffle=True, num_workers=int(self.opt.workers))
        return trainloader,testloader,dataloader,testset