from model import U_net
from dataset import FujiDataset,SonyDataset
import torch.utils.data as Data
from model_utils.common_tools import *
#from utils.data_provider import *
import FCN_solver
import torchvision.transforms as transforms

transform_img = transforms.Compose([
                    transforms.ToTensor()
                    ])
models=[]
models.append(U_net.U_net())

for i in range(0,len(models)):
    models[i]=nn.DataParallel(models[i],device_ids=[0,1])

lrs=[0.0001]

optimizers=generate_optimizers(models,lrs,optimizer_type="adam",weight_decay=0.001)
function=weights_init(init_type='xavier')
solver=FCN_solver.FCN_solver(models,"U_net")
solver.set_optimizers(optimizers)
solver.init_models(function)
solver.cuda()
sony_train_dataset=SonyDataset.SonyDataset(info_path='Sony_test_list.txt',img_path='/home/huangxiaoyu/dataset/',transform=transform_img,patch_size=128)
sony_train_dataloader=Data.DataLoader(dataset=sony_train_dataset,batch_size=16,shuffle=True,num_workers=0)

#train_dataprovider=data_provider(train_provider_dataset ,batch_size=16, is_cuda=False)
#train_dataloader=Data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)
param_dict={}
param_dict["loader"]=sony_train_dataloader
#param_dict["provider"]=train_dataprovider
solver.test_one_batch(param_dict)
