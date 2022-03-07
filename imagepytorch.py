import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from imagedataset import ImageDataset, ImageDataLoader
from imagenet import ImageDiscriminator, ImageGenerator
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
 
class ImagePytorch:
    def __init__(self, gen:ImageGenerator, dis:ImageDiscriminator, z_dim):
        self.gen = gen
        self.dis = dis
        self.z_dim = z_dim
        
        self.g_optimizer = optim.Adam(self.gen.parameters(), 0.0001, [0.0, 0.9])
        self.d_optimizer = optim.Adam(self.dis.parameters(), 0.0004, [0.0, 0.9])
        # self.criteron = nn.BCEWithLogitsLoss(reduction="mean")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device={}".format(self.device))
        
        self.gen.to(self.device)
        self.dis.to(self.device)
    
    def train(self, dataloader:ImageDataLoader, epoch_num):
        self.gen.train()
        self.dis.train()
        
        for epoch in range(1, epoch_num+1):            
            epoch_gen_loss_sum = 0.0
            epoch_dis_loss_sum = 0.0
            
            for real_images, _ in tqdm(dataloader()):
                batch_size = real_images.size()[0]
                # real_label = torch.full((batch_size,), 1.0).to(self.device)
                # fake_label = torch.full((batch_size,), 0.0).to(self.device)
                
                # Discriminatorの学習
                fake_inputs = torch.randn(batch_size, self.z_dim)
                fake_inputs = fake_inputs.view(batch_size, self.z_dim, 1, 1)
                fake_inputs = fake_inputs.to(self.device)                
                
                real_images = real_images.to(self.device)
                fake_images = self.gen(fake_inputs)
                               
                d_out_real = self.dis(real_images)
                d_out_fake = self.dis(fake_images)
                
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                # d_loss_real = self.criteron(d_out_real.view(-1), real_label)
                # d_loss_fake = self.criteron(d_out_fake.view(-1), fake_label)
                d_loss = d_loss_real + d_loss_fake
                
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                
                d_loss.backward()
                self.d_optimizer.step()
                
                # Generatorの学習
                fake_inputs = torch.randn(batch_size, self.z_dim)
                fake_inputs = fake_inputs.view(batch_size, self.z_dim, 1, 1)
                fake_inputs = fake_inputs.to(self.device)
                
                fake_images = self.gen(fake_inputs)
                
                d_out_fake = self.dis(fake_images)
                
                g_loss = - d_out_fake.mean()
                # g_loss = self.criteron(d_out_fake.view(-1), real_label)
                
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # loss記録
                epoch_dis_loss_sum += d_loss.item()
                epoch_gen_loss_sum += g_loss.item()
        
            print("epoch_num={}, g_loss={:.4f}, d_loss={:.4f}".format(epoch, epoch_gen_loss_sum / dataloader.batch_size, epoch_dis_loss_sum / dataloader.batch_size))
                
    def predict(self, generate_num, dataloader: ImageDataLoader):
        fake_inputs = torch.randn(generate_num, self.z_dim)
        fake_inputs = fake_inputs.view(generate_num, self.z_dim, 1, 1)
        fake_inputs = fake_inputs.to(self.device)
                        
        self.gen.eval()
        self.dis.eval()
        fake_images = self.gen(fake_inputs)
        dis_outs = self.dis(fake_images)
        for fake_img, dis_out in zip(fake_images, dis_outs):
            print("dis_out={:.4f}".format(dis_out[0][0][0]))
            fake_img = torch.nn.ReLU()(fake_img.permute(1, 2, 0))
            img = fake_img.cpu().detach().numpy()
            img = (img * 255).astype(np.int)
            plt.imshow(img)
            plt.show()
        
        real_images, _ = next(iter(dataloader()))
        real_images = real_images.to(self.device)
        dis_outs = self.dis(real_images)
        for i, (real_img, dis_out) in enumerate(zip(real_images, dis_outs)):
            if i >= generate_num:
                break
            
            print("dis_out={:.4f}".format(dis_out[0][0][0]))
            real_img = real_img.permute(1, 2, 0)
            img = real_img.cpu().detach().numpy()
            plt.imshow(img)
            plt.show()
            
if __name__=="__main__":
    image_gen = ImageGenerator(3, 32, 32, 20)
    image_dis = ImageDiscriminator(3, 32, 32)
    train = ImagePytorch(image_gen, image_dis, 20)
    
    datapath=""
    if len(sys.argv) == 4:
        datapath=sys.argv[1]
        gen_weight_path = os.path.join(sys.argv[2], "gen_weight.dat")
        dis_weight_path = os.path.join(sys.argv[2], "dis_weight.dat")
        epoch_num = int(sys.argv[3])
    else:
        # datapath=r"D:\git\pytorch_image2\data\cifar10_small\data"
        # datapath=r"D:\git\pytorch_image2\data\cifar10\cifar10_data\train"
        # datapath=r"D:\git\pytorch_image2\data\number\data"
        datapath=r"D:\git\pytorch_image2\data\cifar10_airplane"
        gen_weight_path = r"D:\git\pytorch_image2\savedir\gen_weight.dat"
        dis_weight_path = r"D:\git\pytorch_image2\savedir\dis_weight.dat"
        
        if len(sys.argv) == 2:
            epoch_num = int(sys.argv[1])
        else:
            epoch_num = 2
           
    dataset = ImageDataset(True, 32)
    dataset.load_numpys(datapath)
    dataloader = ImageDataLoader(dataset, 50)
    
    if os.path.isfile(gen_weight_path):
        image_gen.load_weight(gen_weight_path)
    
    if os.path.isfile(dis_weight_path):
        image_dis.load_weight(dis_weight_path)
    
    if epoch_num > 0:    
        train.train(dataloader, epoch_num)
        
    image_gen.save_weight(gen_weight_path)
    image_dis.save_weight(dis_weight_path)
    
    train.predict(5, dataloader)
    
    