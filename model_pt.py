import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(Upsample, self).__init__()
        self.in_channels = input_shape[0]
        self.width = input_shape[1]
        self.height = input_shape[2]

        # group 1
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # group 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # group 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # group 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # group 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # group 6

        n_size = self._get_conv_output(input_shape)
        self.fc6 = nn.Linear(in_features=n_size, out_features=512)
        self.drop6 = nn.Dropout2d(p=0.5, inplace=True)

        # group7
        self.fc7 = nn.Linear(in_features=512, out_features=512)
        self.drop7 = nn.Dropout2d(p=0.5, inplace=True)

        # output
        self.fc8 = nn.Linear(in_features=512, out_features=33 * 12 * 5)

        self.bn_pool4 = nn.BatchNorm2d(num_features=33)
        self.conv4 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=3, padding=1)
        self.bn_pool3 = nn.BatchNorm2d(num_features=33)
        self.conv3 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=3, padding=1)
        self.bn_pool2 = nn.BatchNorm2d(num_features=33)
        self.conv2 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=3, padding=1)
        self.bn_pool1 = nn.BatchNorm2d(num_features=33)
        self.conv1 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=3, padding=1)

        # deconvolution
        scale = 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=1, stride=1, padding=0,
                                          bias=False)
        scale *= 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=(2 * scale), stride=scale,
                                          padding=int(scale / 2), bias=False)
        scale *= 2
        self.deconv3 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=(2 * scale), stride=scale,
                                          padding=int(scale / 2), bias=False)
        scale *= 2
        self.deconv4 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=(2 * scale), stride=scale,
                                          padding=int(scale / 2), bias=False)
        scale *= 2
        self.deconv5 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=(2 * scale), stride=scale,
                                          padding=int(scale / 2), bias=False)

        scale = 2
        self.deconv6 = nn.ConvTranspose2d(in_channels=33, out_channels=33, kernel_size=(2 * scale), stride=scale,
                                          padding=int(scale / 2), bias=False)
        self.conv6 = nn.Conv2d(in_channels=33, out_channels=33, kernel_size=3, padding=1)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5(x)
        
        return x

    def forward(self, x):
        batch_size = x.size(0)

        x = self._forward_features(x)
        
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        x = self.drop7(x)
        x = self.fc8(x)

        x = x.view(batch_size, 33, 5, 12)
        x = self.bn_pool4(x)
        x = self.conv4(x)
        x = self.bn_pool3(x)
        x = self.conv3(x)
        x = self.bn_pool2(x)
        x = self.conv2(x)
        x = self.bn_pool1(x)
        x = F.relu(self.conv1(x))

        pred1 = F.relu(self.deconv1(x))
        pred2 = F.relu(self.deconv2(pred1))
        pred3 = F.relu(self.deconv3(pred2))
        pred4 = F.relu(self.deconv4(pred3))
        pred5 = F.relu(self.deconv5(pred4))

        feats = pred1 + pred2 + pred3 + pred4 + pred5
        feats = F.relu(feats)

        up = F.relu(self.deconv6(feats))
        up = self.conv6(up)

        return up


if __name__ == '__main__':
    model = Upsample(input_shape=(3, 32, 32))
    print(model)
    input_var = Variable(torch.randn(1, 3, 32, 32))
    model.cuda()
    out_var = model(input_var.cuda())
    
