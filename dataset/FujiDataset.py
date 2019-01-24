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

class FujiDataset(Data.Dataset):

    def __init__(self,info_path='Fuji_train.txt',img_path='/home/huangxiaoyu/dataset/SID/',transform=transform_img,patch_size=256):
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


    #pack X-Trans image to 9 channels
    def pack_raw(self,im):
        #im=np.expand_dims(im,axis=2)
        img_shape=im.shape
        H=(img_shape[0]//6)*6
        W=(img_shape[1]//6)*6

        out=np.zeros((H//3,W//3,9))

        #0 R
        out[0::2,0::2,0]=im[0:H:6,0:W:6]
        out[0::2,1::2,0]=im[0:H:6,4:W:6]
        out[1::2,0::2,0]=im[3:H:6,1:W:6]
        out[1::2,1::2,0]=im[3:H:6,3:W:6]

        #1 G
        out[0::2,0::2,1]=im[0:H:6,2:W:6]
        out[0::2,1::2,1]=im[0:H:6,5:W:6]
        out[1::2,0::2,1]=im[3:H:6,2:W:6]
        out[1::2,1::2,1]=im[3:H:6,5:W:6]

        #1 B
        out[0::2,0::2,2]=im[0:H:6,1:W:6]
        out[0::2,1::2,2]=im[0:H:6,3:W:6]
        out[1::2,0::2,2]=im[3:H:6,0:W:6]
        out[1::2,1::2,2]=im[3:H:6,4:W:6]

        #4 R
        out[0::2,0::2,3]=im[1:H:6,2:W:6]
        out[0::2,1::2,3]=im[2:H:6,5:W:6]
        out[1::2,0::2,3]=im[5:H:6,2:W:6]
        out[1::2,1::2,3]=im[4:H:6,5:W:6]

        #5 B
        out[0::2,0::2,4]=im[2:H:6,2:W:6]
        out[0::2,1::2,4]=im[1:H:6,5:W:6]
        out[1::2,0::2,4]=im[4:H:6,2:W:6]
        out[1::2,1::2,4]=im[5:H:6,5:W:6]

        out[:,:,5]=im[1:H:3,0:W:3]
        out[:,:,6]=im[1:H:3,1:W:3]
        out[:,:,7]=im[2:H:3,0:W:3]
        out[:,:,8]=im[2:H:3,1:W:3]

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
        gt_patch=gt_img[yy*3:yy*3+self.patch_size*3,xx*3:xx*3+3*self.patch_size,:]

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

        print(input_patch)
        print(gt_patch)
        return input_patch,gt_patch


if __name__=='__main__':
    fuji_train_dataset=FujiDataset(info_path='Fuji_train.txt',img_path='/home/huangxiaoyu/dataset/SID/',transform=transform_img,patch_size=256)
    fuji_train_dataloader=Data.DataLoader(dataset=fuji_train_dataset,batch_size=2,shuffle=True,num_workers=0)
    for i,(input_patch,gt_patch) in enumerate(fuji_train_dataloader):
        print(input_patch,gt_patch)
        print("train")
        break
