from model import U_net
from dataset import celebadataset
import torch.utils.data as Data
from utils.common_tools import *
from utils.data_provider import *
import FCN_solver

models=[]
models.append(U_net.U_net())

for i in range(0,len(models)):
    models[i]=nn.DataParallel(models[i],device_ids=[0,1])

lrs=[0.0001]

optimizers=generate_optimizers(models,lrs,optimizer_type="adam",weight_decay=0.001)
function=weights_init(init_type='xavier')
solver=FCN_sovler.FCN_solver(models,"U_net")
solver.init_models(function)
solver.cuda()
train_dataset=celebadataset.celebadataset(img_dir='./dataset/img_align_celeba',attrpath='./dataset/list_attr_celeba.txt',identipath='./dataset/identity_CelebA.txt',transform=celebadataset.transform_img,mode='train',load_data=True)
train_provider_dataset=celebadataset.celebadataset(img_dir='./dataset/img_align_celeba',attrpath='./dataset/list_attr_celeba.txt',identipath='./dataset/identity_CelebA.txt',transform=celebadataset.transform_img,mode='train',load_data=True)

train_dataprovider=data_provider(train_provider_dataset ,batch_size=16, is_cuda=False)
train_dataloader=Data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)
param_dict={}
param_dict["loader"]=train_dataloader
param_dict["provider"]=train_dataprovider
solver.train_loop(param_dict,epochs=4000)
