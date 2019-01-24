import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data as Data
import rawpy
try:
    import cPickle as pickle
except ImportError:
    import pickle

transform_img = transforms.Compose([
                    transforms.ToTensor()
                    ])

class SonyDataset(Data.Dataset):

    def __init__(self,info_path='Sony_train.txt',img_path='/home/huangxiaoyu/dataset/SID/',transform=transform_img,patch_size=512):
        self.info_path=info_path
        self.img_path=img_path
        self.transform=transform
        self.alldata=[]
        self.readfile()
        self.patch_size=patch_size


    def readfile(self):
        alldata=[]
        lines = [line.rstrip() for line in open(self.info_path, 'r')]
        for i, line in enumerate(lines):
            split = line.split()
            alldata.append(split)
        self.alldata=alldata

    #pack Bayer image to 4 channels
    def pack_raw(self,im):
        im=np.expand_dims(im,axis=2)
        img_shape=im.shape
        H=img_shape[0]
        W=img_shape[1]

        out=np.concatenate((im[0:H:2,0:W:2,:],
                            im[0:H:2,1:W:2,:],
                            im[1:H:2,1:W:2,:],
                            im[1:H:2,0:W:2,:]),axis=2)
        #print("out",out)
        return out



    def __len__(self):
        return len(self.alldata)


    def __getitem__(self,index):
        in_path=self.alldata[index][0]
        gt_path=self.alldata[index][1]

        #read img
        input_img=self.pack_raw(np.load(self.img_path+in_path))
        gt_img=np.load(self.img_path+gt_path)

        #crop
        H=input_img.shape[0]
        W=input_img.shape[1]


        xx=np.random.randint(0,W-self.patch_size)
        yy=np.random.randint(0,H-self.patch_size)

        input_patch=input_img[yy:yy+self.patch_size,xx:xx+self.patch_size,:]
        gt_patch=gt_img[yy*2:yy*2+self.patch_size*2,xx*2:xx*2+2*self.patch_size,:]

        if(np.random.randint(2,size=1)[0]==1):
            input_patch=np.flip(input_patch,axis=1)
            gt_patch=np.flip(gt_patch,axis=1)
        if(np.random.randint(2,size=1)[0]==1):
            input_patch=np.flip(input_patch,axis=2)
            gt_patch=np.flip(gt_patch,axis=2)
        #if(np.random.randint(2,size=1)[0]==1):
        #    input_patch=np.transpose(input_patch,(0,2,1,3))
        #    gt_patch=np.transpose(gt_patch,(0,2,1,3))

        input_patch=np.minimum(input_patch,1.0)

        if(self.transform!=None):
            input_patch=torch.from_numpy(input_patch.copy())#self.transform(input_patch)
            gt_patch=torch.from_numpy(gt_patch.copy())#self.transform(gt_patch)
            input_patch=input_patch.permute(2,0,1)
            gt_patch=gt_patch.permute(2,0,1)

        return input_patch,gt_patch


if __name__=='__main__':
    sony_train_dataset=SonyDataset(info_path='Sony_train.txt',img_path='/home/huangxiaoyu/dataset/SID/',transform=transform_img,patch_size=256)
    sony_train_dataloader=Data.DataLoader(dataset=sony_train_dataset,batch_size=2,shuffle=True,num_workers=0)
    for i,(input_patch,gt_patch) in enumerate(sony_train_dataloader):
        print(input_patch,gt_patch)
        print("train")
        break
