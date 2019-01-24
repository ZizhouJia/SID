import torch
import torch.nn as nn
import math
import numpy as np

class U_net(nn.Module):
    def __init__(self,input_depth=4,output_depth=12):
        super(U_net,self).__init__()
        self.activate=nn.LeakyReLU(0.2)
        kernels_list=[input_depth,32,64,128,256,512]
        self.down=nn.ModuleList()
        self.randint=np.random.randint(0,100000)
        for i in range(0,4):
            conv1=nn.Conv2d(kernels_list[i],kernels_list[i+1],3,1,1)
            conv2=nn.Conv2d(kernels_list[i+1],kernels_list[i+1],3,1,1)
            max_pool=nn.MaxPool2d(2)
            # self.add_module("conv1_down_"+str(i+self.randint),conv1)
            # self.add_module("conv2_down_"+str(i+self.randint),conv2)
            # self.add_module("max_pool_down_"+str(i+self.randint),max_pool)
            self.down.append(conv1)
            self.down.append(conv2)
            self.down.append(max_pool)

        self.conv_middle1=nn.Conv2d(kernels_list[4],kernels_list[5],3,1,1)
        self.conv_middle2=nn.Conv2d(kernels_list[5],kernels_list[5],3,1,1)

        self.pow=math.sqrt(output_depth/3)
        self.pow=int(self.pow)

        self.up=nn.ModuleList()
        for i in range(0,4):
            deconv1=nn.ConvTranspose2d(kernels_list[5-i],kernels_list[4-i],3,2,padding=1,output_padding=1)
            conv1=nn.Conv2d(kernels_list[4-i]*2,kernels_list[4-i],3,1,1)
            conv2=nn.Conv2d(kernels_list[4-i],kernels_list[4-i],3,1,1)
            # self.add_module("deconv1_up_"+str(3-i+self.randint),deconv1)
            # self.add_module("conv1_up_"+str(3-i+self.randint),conv1)
            # self.add_module("conv2_up_"+str(3-i+self.randint),conv2)
            self.up.append(deconv1)
            self.up.append(conv1)
            self.up.append(conv2)

        self.conv_out=nn.Conv2d(kernels_list[1],output_depth,3,1,1)

    def forward(self,x):
        conv_out=[]
        out=x
        #print(x)
        for i in range(0,4):
            out=self.down[i*3+0](out)
            out=self.activate(out)
            out=self.down[i*3+1](out)
            out=self.activate(out)
            conv_out.append(out)
            out=self.down[i*3+2](out)
            out=self.activate(out)
        out=self.conv_middle1(out)
        out=self.activate(out)
        out=self.conv_middle2(out)
        out=self.activate(out)
        for i in range(0,4):
            out=self.up[3*i+0](out)
            out=self.activate(out)
            out=torch.cat((out,conv_out[3-i]),1)
            out=self.up[3*i+1](out)
            out=self.activate(out)
            out=self.up[3*i+2](out)
            out=self.activate(out)
        out=self.conv_out(out)
        out=out.permute(0,2,3,1)
        H=out.size()[1]
        W=out.size()[2]
        out=out.contiguous().view(-1,1,H*W,self.pow*self.pow*3)
        out=out.contiguous().view(-1,1,H*W*self.pow,self.pow*3)
        out=out.contiguous().view(-1,H*self.pow,W,self.pow*3)
        out=out.contiguous().view(-1,H*self.pow,1,W*self.pow*3)
        out=out.contiguous().view(-1,H*self.pow,W*self.pow,3)
        out=out.permute(0,3,1,2)
        return out
