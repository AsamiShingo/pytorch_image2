from matplotlib.pyplot import xcorr
from sklearn.cluster import k_means
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import math

class ImageSelfAttention(nn.Module):
    def __init__(self, in_dim, conv_dim):
        super().__init__()
        self.query_conv_layer = nn.Conv2d(in_dim, conv_dim, kernel_size=1)
        self.key_conv_layer = nn.Conv2d(in_dim, conv_dim, kernel_size=1)
        self.value_conv_layer = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.soft_max = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        data_size = input.shape[2]*input.shape[3]
        query_data = self.query_conv_layer(input).view(input.shape[0], -1, data_size)
        key_data = self.key_conv_layer(input).view(input.shape[0], -1, data_size)
        value_data = self.value_conv_layer(input).view(input.shape[0], -1, data_size)
        
        attention_matrix = torch.bmm(query_data.permute(0, 2, 1), key_data)
        attention_score_t = self.soft_max(attention_matrix)
        attention_data = torch.bmm(value_data, attention_score_t)
        attention_data = attention_data.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        
        attention_result = input + self.gamma * attention_data
        
        return attention_result
        
class ImageGenerator(nn.Module):
    def __init__(self, chanel, width, height, z_dim=20):
        super().__init__()
        self.chanel = chanel
        self.width = width
        self.height = height
        self.z_dim = z_dim
        self.image_size = int(math.sqrt(self.width * self.height)) * self.chanel
        self.layers = []
        
        c = self.z_dim
        
        def create_transpose_layer(in_c, out_c, kernel, stride, padding):
            layers = []
            layers.append(nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding)))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
                
            return layers, out_c
        
        layers_tmp, c = create_transpose_layer(c, self.chanel*self.image_size*4, 4, 1, 0)
        self.layers += layers_tmp
        layers_tmp, c = create_transpose_layer(c, self.chanel*self.image_size*2, 4, 2, 1)
        self.layers += layers_tmp
        self.layers.append(ImageSelfAttention(c, c//8))
        layers_tmp, c = create_transpose_layer(c, self.chanel*self.image_size*1, 4, 2, 1)
        self.layers += layers_tmp
        self.layers.append(ImageSelfAttention(c, c//8))
        
        self.layers.append(nn.ConvTranspose2d(c, self.chanel, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Tanh())
        
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, input):
        x = input
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def init_weight(self):
        def init_weight_func(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
        super().apply(init_weight_func)                
                
    def save_weight(self, path):
        torch.save(super().state_dict(), path)
        
    def load_weight(self, path):
        try:
            weights = torch.load(path)
        except:
            weights = torch.load(path, map_location={'cuda:0': 'cpu'})
            
        super().load_state_dict(weights)
        
        
class ImageDiscriminator(nn.Module):    
    def __init__(self, chanel, width, height):
        super().__init__()
        self.chanel = chanel
        self.width = width
        self.height = height
        self.image_size = int(math.sqrt(self.width * self.height)) * self.chanel
        self.layers = []
        
        c = self.chanel
        
        def create_conv_layer(in_c, out_c, kernel, stride, padding):
            layers = []            
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding)))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
                
            return layers, out_c
        
        layers_tmp, c = create_conv_layer(c, self.image_size*1, 4, 2, 1)
        self.layers += layers_tmp
        layers_tmp, c = create_conv_layer(c, self.image_size*2, 4, 2, 1)
        self.layers += layers_tmp
        self.layers.append(ImageSelfAttention(c, c//8))
        layers_tmp, c = create_conv_layer(c, self.image_size*4, 4, 2, 1)
        self.layers += layers_tmp
        self.layers.append(ImageSelfAttention(c, c//8))
                
        self.layers.append(nn.Conv2d(c, 1, kernel_size=4, stride=1, padding=0))
        # self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, input):
        x = input
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def init_weight(self):
        def init_weight_func(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
        super().apply(init_weight_func)   
        
    def save_weight(self, path):
        torch.save(super().state_dict(), path)
        
    def load_weight(self, path):
        try:
            weights = torch.load(path)
        except:
            weights = torch.load(path, map_location={'cuda:0': 'cpu'})
            
        super().load_state_dict(weights)
        
        
if __name__=="__main__":
    image_gen = ImageGenerator(1, 32, 32, 20)    
    print(image_gen)
    
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    
    gen_out = image_gen(input_z)
    print(gen_out.shape)
    print(gen_out)
    # img = fake_images[0][0].detach().numpy()
    # print(img.shape)
    # plt.imshow(img, 'gray')
    # plt.show()
    
    image_dis = ImageDiscriminator(1, 32, 32)
    print(image_dis)
    
    dis_out = image_dis(gen_out)
    print(dis_out.shape)
    print(dis_out)