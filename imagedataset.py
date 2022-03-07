from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import numpy as np
import cv2

class ImageDataset(Dataset):
    NUMPY_EXT=".npz"
    IMAGE_EXT=".bmp"
    NUMPY_KEY="image"
    
    TRANSFORM_SCALE=(0.5, 1.0)
    # TRANSFORM_MEAN=(0.485, 0.456, 0.406)
    # TRANSFORM_STD=(0.229, 0.224, 0.225)
    
    def __init__(self, is_train, transform_size):
        self.is_train = is_train
        self.transform_size = transform_size
        
        self.transform = self._get_transform(self.is_train, self.transform_size)
        self.imglist = []
        self.labellist = []
        self.labelnames = {}
        
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, index):
        return self.transform(self.imglist[index]), torch.from_numpy(self.labellist[index].astype(np.float32)).clone()
        
    def _get_transform(self, is_train, transform_size):
        transform = None
        
        if(is_train == True):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(),
				# transforms.RandomResizedCrop(transform_size, scale=self.TRANSFORM_SCALE),
				# transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				# transforms.Normalize(self.TRANSFORM_MEAN, self.TRANSFORM_STD)    
			])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(),
				# transforms.Resize(transform_size),
				# transforms.CenterCrop(transform_size),
				transforms.ToTensor(),
				# transforms.Normalize(self.TRANSFORM_MEAN, self.TRANSFORM_STD)    
			])
            
        return transform
            
    def load_numpys(self, datas_path):
        self.imglist = []
        self.labellist = []
        self.labelnames = {}
        
        data_files = []
        for data_file in os.listdir(datas_path):
            data_path = os.path.join(datas_path, data_file)
            if os.path.isfile(data_path):
                data_files.append(data_file)
                
        for data_file in data_files:
            data_path = os.path.join(datas_path, data_file)
            print("load file(file={})".format(data_path))
            datas_np = np.load(data_path)
            label_id = data_file.split("_")[0]
            label_name = data_file.split("_")[1]
            self.labelnames[label_id] = label_name
            
            label_vector = np.eye(len(data_files))[int(label_id)]
            for data in datas_np[self.NUMPY_KEY]:
                self.imglist.append(data)
                self.labellist.append(label_vector)
    
    @classmethod                
    def images_to_numpys(cls, images_path, datas_path, width, height):
        images = []
        
        os.makedirs(datas_path, exist_ok=True)

        #データロード
        for image_dir in os.listdir(images_path):
            images = []
            
            image_path = os.path.join(images_path, image_dir)
            if os.path.isdir(image_path):
                print("load directory(dir={})".format(image_path))                
                images = cls.__images_to_numpys_sub(image_path, images, width, height)
        
            images_np =  np.array(images)
            #圧縮して保存
            np.savez_compressed(os.path.join(datas_path, image_dir + cls.NUMPY_EXT), image=images_np)

    @classmethod 
    def __images_to_numpys_sub(cls, path, images, width, height):
        for imagefile in os.listdir(path):
            imagepath = os.path.join(path, imagefile)
            if os.path.isfile(imagepath):
                image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
                
                if width != image.shape[1] or height != image.shape[0]:
                    image = cv2.resize(image , (width, height))
                    
                images.append(image)
            else:
                images = cls.__images_to_numpys_sub(imagepath, images, width, height)
        
        return images

    @classmethod 
    def numpys_to_images(cls, datas_path, images_path):
        #フォルダ作成
        os.makedirs(images_path, exist_ok=True)
  
        for data_file in os.listdir(datas_path):            
            data_path = os.path.join(datas_path, data_file)
            if os.path.isfile(data_path):
                print("load file(file={})".format(data_path))
                datas_np = np.load(data_path)
                image_path = os.path.join(images_path, data_file.split(".")[0])
                os.makedirs(image_path, exist_ok=True)
                
                #画像保存
                for i, data in enumerate(datas_np[cls.NUMPY_KEY]):
                    filename = os.path.join(image_path, str(i).zfill(8) + cls.IMAGE_EXT)
                    cv2.imwrite(filename, data)

class ImageDataLoader:
    def __init__(self, dataset:ImageDataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __call__(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.dataset.is_train)
        
if __name__=="__main__":
    ImageDataset.images_to_numpys(r"D:\git\pytorch_image2\data\number\image", r"D:\git\pytorch_image2\data\number\data", 32, 32)
    # ImageDataset.images_to_numpys(r"D:\git\pytorch_image2\data\cifar10_small\image", r"D:\git\pytorch_image2\data\cifar10_small\data", 32, 32)
    # ImageDataset.numpys_to_images(r"D:\git\pytorch_test\cifar10\cifar10_data\train", r"D:\git\pytorch_test\cifar10\cifar10_image_out\train")