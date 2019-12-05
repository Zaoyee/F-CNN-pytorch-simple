import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn

class fcn_32(nn.Module):
    def __init__(self, num_classes=21):
        super(fcn_32, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet34.children())[:-4])
        self.layer2 = list(resnet34.children())[-4]
        self.layer3 = list(resnet34.children())[-3]

        self.conv1 = nn.Conv2d(512, num_classes, 1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        self.conv3 = nn.Conv2d(128, num_classes, 1)


        self.upsample_32x = nn.ConvTranspose2d(num_classes, num_classes, 64, 32, 16, bias=False)
        self.upsample_32x.weight.data = self.bilinear_kernal(num_classes, num_classes, 64)

        #self.unsmaple_32x = nn.ConvTranspose2d(num_classes, num_classes,  )

    def bilinear_kernal(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        # return open multi-dimensional meshgrid
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

    def forward(self, x):
        # 1/8
        print(x.shape)
        x = self.layer1(x)

        # 1/16
        x = self.layer2(x)
        cv2 = x

        # 1/32
        x = self.layer3(x)
        cv3 = x

        cv3 = self.conv1(cv3)
        print(cv3.shape)
        out1 = self.unsample_32x(cv3)

        return out1[:,:,:,1:-1]