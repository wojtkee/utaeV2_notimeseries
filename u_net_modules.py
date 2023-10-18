import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UNET_v3(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(UNET_v3,self).__init__()

        # down layers

        self.input_conv=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        
        self.down_conv1=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(64,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True))
        
        self.down_conv2=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(128,256,kernel_size=3,padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        
        self.down_conv3=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(256,512,kernel_size=3,padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True))
        

        self.down_conv4=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(512,1024,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True))


        # up layers

        self.up_conv41=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv31=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up_conv21=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up_conv11=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)


        self.up_conv42=nn.Sequential(nn.Conv2d(1024,512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        
        self.up_conv32=nn.Sequential(nn.Conv2d(512,256, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
    
        self.up_conv22=nn.Sequential(nn.Conv2d(256,128, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        
        self.up_conv12=nn.Sequential(nn.Conv2d(128,64, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        

        self.out_conv=nn.Conv2d(64,out_channels,kernel_size=1)


    def forward(self,x):

        # przeplyw down

        
        xinput = self.input_conv(x)
        # print("xin shape:",xinput.shape)
        x1 = self.down_conv1(xinput)
        # print("x1 shape:",x1.shape)
        x2 = self.down_conv2(x1)
        # print("x2 shape:",x2.shape)
        x3 = self.down_conv3(x2)
        # print("x3 shape:",x3.shape)
        x4 = self.down_conv4(x3)
        


   

        # przeplyw up

        x = self.up_conv41(x4)
        # print("x:",x.shape)
        # print("x3",x3.shape)
        x=torch.cat([x3,x],dim=1)
        # print(x.shape)
        x = self.up_conv42(x)





        x=self.up_conv31(x)
        # print('x_1',x.shape)
        # x = self.up_conv21(x)
        # print('x_2',x.shape)
        x=torch.cat([x2,x],dim=1)
        x=self.up_conv32(x)

        # print('xx2', x.shape)






        x = self.up_conv21(x)
        x=torch.cat([x1,x],dim=1)
        x=self.up_conv22(x)

        
        # print('xx1',x.shape)


        x=self.up_conv11(x)
        x=torch.cat([xinput,x],dim=1)
        x=self.up_conv12(x)



        # print('after out',x.shape)
        log=self.out_conv(x)
        # print('out',log.shape)





        _,preds=torch.max(log,dim=1)
        preds=preds.to(torch.float32)
        log=log.to(torch.float32)

        # print('predictions size:',preds.shape)

        return log, preds
    




class UNET_v3_bilstm(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(UNET_v3,self).__init__()

        # down layers

        self.input_conv=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        
        self.down_conv1=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(64,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True))
        
        self.down_conv2=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(128,256,kernel_size=3,padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        
        self.down_conv3=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(256,512,kernel_size=3,padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True))
        

        self.down_conv4=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(512,1024,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True))


        # up layers

        self.up_conv41=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv31=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up_conv21=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up_conv11=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)


        self.up_conv42=nn.Sequential(nn.Conv2d(1024,512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        
        self.up_conv32=nn.Sequential(nn.Conv2d(512,256, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
    
        self.up_conv22=nn.Sequential(nn.Conv2d(256,128, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        
        self.up_conv12=nn.Sequential(nn.Conv2d(128,64, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        

        self.out_conv=nn.Conv2d(64,out_channels,kernel_size=1)



        self.blstm=nn.LSTM(
            input_size=64,
        )


    def forward(self,x):

        # przeplyw down

        
        xinput = self.input_conv(x)
        # print("xin shape:",xinput.shape)
        x1 = self.down_conv1(xinput)
        # print("x1 shape:",x1.shape)
        x2 = self.down_conv2(x1)
        # print("x2 shape:",x2.shape)
        x3 = self.down_conv3(x2)
        # print("x3 shape:",x3.shape)
        x4 = self.down_conv4(x3)
        


   

        # przeplyw up

        x = self.up_conv41(x4)
        # print("x:",x.shape)
        # print("x3",x3.shape)
        x=torch.cat([x3,x],dim=1)
        # print(x.shape)
        x = self.up_conv42(x)





        x=self.up_conv31(x)
        # print('x_1',x.shape)
        # x = self.up_conv21(x)
        # print('x_2',x.shape)
        x=torch.cat([x2,x],dim=1)
        x=self.up_conv32(x)

        # print('xx2', x.shape)






        x = self.up_conv21(x)
        x=torch.cat([x1,x],dim=1)
        x=self.up_conv22(x)

        
        # print('xx1',x.shape)


        x=self.up_conv11(x)
        x=torch.cat([xinput,x],dim=1)
        x=self.up_conv12(x)



        # print('after out',x.shape)
        log=self.out_conv(x)
        # print('out',log.shape)





        _,preds=torch.max(log,dim=1)
        preds=preds.to(torch.float32)
        log=log.to(torch.float32)

        # print('predictions size:',preds.shape)

        return log, preds