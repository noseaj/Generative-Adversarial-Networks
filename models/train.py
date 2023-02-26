import torch
import torchvision.utils as vutils
import time
from results.fid_Scoring import FID

class Train():
    def __init__(self, opt, netD, netG, trainloader, testloader, criterion, optimizerD, optimizerG, testset):
        self.opt = opt
        self.netD = netD
        self.netG = netG
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.testset = testset

    def Train(self, device):
        # writer = SummaryWriter()
        fixed_noise = torch.randn(self.opt.batchSize, self.opt.nz, 1, 1, device=device)
        fid_obj = FID()

        for epoch in range(self.opt.niter): #for each epoch
            start = time.time()
            for i, data in enumerate(self.trainloader, 0): #for each batch of trainset using trainloader
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train discriminator with real
                self.netD.zero_grad() #clear grad data for this iteration
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((self.opt.batchSize,), self.opt.real_label, dtype=torch.float, device=device) #label 1

                output_D = self.netD(real_cpu).view(-1) #get discriminator prediction
                errD_real = self.criterion(output_D, label) #apply BCE loss function on output with real label values
                errD_real.backward() #back propagate the loss in discriminator 
                D_x = output_D.mean().item() #final discriminator loss on real data

                # train with fake
                noise = torch.randn(self.opt.batchSize, self.opt.nz, 1, 1, device=device)
                fake = self.netG(noise) #generate noise data
                label.fill_(self.opt.fake_label) 
                output = self.netD(fake.detach()).view(-1) #get discriminator prediction
                errD_fake = self.criterion(output, label) #apply BCE loss function
                errD_fake.backward() #back propagate the loss in discriminator
                D_G_z1 = output.mean().item() #final discriminator loss on fake data
                errD = errD_real + errD_fake #final discriminator error
                self.optimizerD.step() #apply Adam optimizer

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.opt.real_label)  # fake labels are real for generator cost
                output = self.netD(fake).view(-1) #prediction of discriminator
                errG = self.criterion(output, label) #generator loss function
                errG.backward() #back propagate in generator
                D_G_z2 = output.mean().item() 
                self.optimizerG.step() #use adam optimizer

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, self.opt.niter, i, len(self.trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with open('log\\train_info.txt','a') as f:
                    f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'% (epoch, self.opt.niter, i, len(self.trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)) 
                    f.write('\n')
                #end of epoch operations
                if i == len(self.trainloader)-1: #if last iteration in epoch
                #     vutils.save_image(real_cpu, #save image of real sample
                #             '%s/real_samples.png' %self.opt.outf,
                #             normalize=True)
                #     fake = self.netG(fixed_noise) #generate fake samples using generator
                #     vutils.save_image(fake.detach(), #save fake images generated
                #             '%s/fake_samples_epoch_%03d.png' % (self.opt.outf, epoch),
                #             normalize=True)

                    #find FID
                    # Number of images to compare for calculating FID score
                    comparison_size = 1000
                    #fake images for FID score
                    noise = torch.randn(comparison_size, self.opt.nz, 1, 1, device=device)
                    fake_data = self.netG(noise)

                    #real images picked randomly from test data for FID score
                    real_data = None
                    rand_sampler = torch.utils.data.RandomSampler(self.testset, num_samples=comparison_size, replacement=True)
                    test_sampler = torch.utils.data.DataLoader(self.testset, batch_size=comparison_size, sampler=rand_sampler)
                    for i,data in enumerate(test_sampler, 0):
                        real_data = data[0]
                        break
                    #computing FID score
                    fid_val = fid_obj.compute_fid(real_data, fake_data)
                    print(fid_val)

                    if epoch % 5 == 0: #save generator and discriminator states once every 50 epochs
                    # do checkpointing
                        path = 'models\\'
                        torch.save(self.netG.state_dict(), f'{path}Generator_{epoch}.pt')
                        #torch.save(self.netD.state_dict(), f'{path}Discrimnator_{epoch}.pt')



            print('Time per epoch: ', time.time()-start)

