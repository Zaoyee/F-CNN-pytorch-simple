import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn

class fcn(nn.Module):
    def __init__(self, num_classes=21):
        super(fcn, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet34.children())[:-4])
        self.layer2 = list(resnet34.children())[-4]
        self.layer3 = list(resnet34.children())[-3]

        self.conv1 = nn.Conv2d(512, num_classes, 1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        self.conv3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.bilinear_kernal(num_classes, num_classes, 16)

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = self.bilinear_kernal(num_classes, num_classes, 4)

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_2x.weight.data = self.bilinear_kernal(num_classes, num_classes, 3)

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
        x = self.layer1(x)
        cv1 = x

        # 1/16
        x = self.layer2(x)
        cv2 = x

        # 1/32
        x = self.layer3(x)
        cv3 = x

        cv3 = self.conv1(cv3)
        #FCN-32
        #out3 = self.unsmaple_32x(cv3)
#         print('cv3 {}'.format(cv3.shape))
        cv3_out1 = self.upsample_2x(cv3)

        cv2 = self.conv2(cv2)
#         print('cv2 {}'.format(cv2.shape))
#         print('cv3out {}'.format(cv3_out1.shape))
        cv2_out1 = cv2 + cv3_out1

        #FCN-16
        #out2 = self.unsmaple_18x(cv2_out1)

        cv1 = self.conv3(cv1)
        cv2_out1 = self.upsample_4x(cv2_out1)
        out1 = cv2_out1 + cv1

        #FCN-8
        out1 = self.upsample_8x(out1)
        return out1[:,:,:,1:-1]
