import rawpy
import torch
import numpy as np
import os


def preprocess_short(raw,ratio,type):
    im=raw.raw_image_visible.astype(np.float32)
    if(type=='Sony'):
        im=ratio*np.maximum(im-512,0)/float(16383-512)
    else:
        im=ratio*np.maximum(im-1024,0)/float(16383-1024)
    return im

def preprocess_long(gt_raw):
    #gt_raw=rawpy.imread(os.path.join(self.img_path,gt_path))
    im=gt_raw.postprocess(use_camera_wb=True,half_size=False,no_auto_bright=True,output_bps=16)
    gt_img=np.float32(im/65535.0)
    return gt_img

def readfile(img_path='/home/huangxiaoyu/dataset/',info_path='Sony_train_list.txt',save_path='/home/huangxiaoyu/dataset/SID/',type='Sony',split='train'):
    #alldata=[]
    #identity
    #imgidentity=[]
    lines = [line.rstrip() for line in open(img_path+info_path, 'r')]
    f=open(type+"_"+split+".txt","w")
    for i, line in enumerate(lines):
        splits = line.split()
        in_path=splits[0]
        gt_path=splits[1]
        iso=splits[2][3::]
        #f=splits[3][1::]

        in_exp=float(in_path[22:-5])
        gt_exp=float(gt_path[21:-5])
        ratio=min(gt_exp/in_exp,300)

        in_raw=rawpy.imread(os.path.join(img_path,in_path))
        gt_raw=rawpy.imread(os.path.join(img_path,gt_path))

        in_img=preprocess_short(in_raw,ratio,type)
        gt_img=preprocess_long(gt_raw)

        H=in_img.shape[0]
        W=in_img.shape[1]

        if(type=='Sony'):
            #4240*2832
            for j in range(3):
                for k in range(4):
                    end_row=min((j+1)*1000,H)
                    end_col=min((k+1)*1000,W)
                    in_split=in_img[j*1000:end_row,k*1000:end_col]
                    gt_split=gt_img[j*1000:end_row,k*1000:end_col]
                    in_split_path=type+"/short/"+split+"_"+str(i)+"_"+str(j*4+k)+".npy"
                    gt_split_path=type+"/long/"+split+"_"+str(i)+"_"+str(j*4+k)+".npy"
                    np.save(save_path+in_split_path,in_split)
                    np.save(save_path+gt_split_path,gt_split)
                    f.write(in_split_path+" "+gt_split_path+"\n")
        else:
            #6000*4000
            for j in range(4):
                for k in range(6):
                    end_row=min((j+1)*1000,H)
                    end_col=min((k+1)*1000,W)
                    in_split=in_img[j*1000:end_row,k*1000:end_col]
                    gt_split=gt_img[j*1000:end_row,k*1000:end_col]
                    in_split_path=type+"/short/"+split+"_"+str(i)+"_"+str(j*6+k)+".npy"
                    gt_split_path=type+"/long/"+split+"_"+str(i)+"_"+str(j*6+k)+".npy"
                    np.save(save_path+in_split_path,in_split)
                    np.save(save_path+gt_split_path,gt_split)
                    f.write(in_split_path+" "+gt_split_path+"\n")
        print("preprocess "+str(i)+" image!")
    f.close()


if __name__=="__main__":
    info_path='Fuji_train_list.txt'
    img_path='/home/huangxiaoyu/dataset/'
    readfile(img_path,info_path,save_path='/home/huangxiaoyu/dataset/SID/',type='Fuji',split='train')
