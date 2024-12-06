from torch import nn
from torch.nn import init

from Classification_three_dimention.NetWork import Encoder, KAN


class Fusion_Network_fuse_other(nn.Module):
    def __init__(self,modelName):
        super(Fusion_Network_fuse_other, self).__init__()

        self.resnet_backbone1=Encoder(modelName,False)
        self.resnet_backbone2=Encoder(modelName,False)
        self.resnet_backbone3=Encoder(modelName,False)
        self.resnet_backbone4=Encoder(modelName,False)
        self.resnet_backbone5=Encoder(modelName,False)

        # self.resnet_backbone=Encoder(modelName,False)

        self.convOut1=nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) ,
            nn.Flatten(),# output size = (1, 1)
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.convOut2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),# output size = (1, 1)
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.convOut3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),# output size = (1, 1)
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.convOut4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),# output size = (1, 1)
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.convOut5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # output size = (1, 1)
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.LinearHead=KAN([256,3])
        # self.LinearHead1 = nn.Sequential(
        #     nn.Linear(256, 3),
        # )
        self.kan1=KAN([256,3])
        self.kan2=KAN([256,3])
        self.kan3=KAN([256,3])
        self.kan4=KAN([256,3])

        self.weights = nn.Parameter(torch.randn(4))
        self.initialize_weights()

    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self,x):

        FA=x[:,0:1,:,:]
        MD=x[:,1:2,:,:]
        AD=x[:,2:3,:,:]
        RD=x[:,3:4,:,:]
        B0=x[:,4:5,:,:]

        out1=self.resnet_backbone1(FA)
        out2=self.resnet_backbone2(MD)
        out3=self.resnet_backbone3(AD)
        out4=self.resnet_backbone4(RD)
        out5=self.resnet_backbone5(B0)

        FA_B0 = attention_fusion_weight(out1, out5, p_type='attention_avg')
        MD_B0 = attention_fusion_weight(out2, out5, p_type='attention_avg')
        AD_B0 = attention_fusion_weight(out3, out5, p_type='attention_avg')
        RD_B0 = attention_fusion_weight(out4, out5, p_type='attention_avg')


        out1 = self.convOut1(FA_B0)
        out2 = self.convOut2(MD_B0)
        out3 = self.convOut3(AD_B0)
        out4 = self.convOut4(RD_B0)



        # 计算平均值
        output_avg = (out1 + out2 + out3 + out4) / 4
        # output = torch.concat([out1, out2, out3, out4], dim=1)

        out1 = self.kan1(out1)
        out2 = self.kan2(out2)
        out3 = self.kan3(out3)
        out4 = self.kan4(out4)
        # 通过线性层
        out_linear_avg = self.LinearHead(output_avg)
        # out_linear_avg = self.LinearHead(output)

        return out_linear_avg, out1, out2, out3, out4
