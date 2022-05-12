'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
# import torch.nn.functional as F

def make_convolution_layers(in_channels, v, batch_norm=False):
    layers = []
    padding = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)
    conv3d = nn.Conv3d(in_channels, v, kernel_size=2, padding=0)
    # layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
    layers += [padding, conv3d, nn.ReLU(inplace=True)]	
    return nn.Sequential(*layers)

class CNN_V1(nn.Module):
    def __init__(self, temp1, temp2, temp3):
        super(CNN_V1, self).__init__()
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.conv_layer_1 = make_convolution_layers(1, 64)
        self.pool_layer_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_layer_2 = make_convolution_layers(64, 128)
        self.pool_layer_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        linear_number = 128
        if temp1[0] != 1 or temp2[0] != 1 or temp3[0] != 1:
            linear_number = 256
            self.conv_layer_3 = make_convolution_layers(128, 256)
            self.conv_layer_4 = make_convolution_layers(256, 256)
            self.pool_layer_3 = nn.MaxPool3d((temp1[0], temp2[0], temp3[0]), stride=2)

        if temp1[1] != 1 or temp2[1] != 1 or temp3[1] != 1:
            linear_number = 256
            self.conv_layer_5 = make_convolution_layers(256, 256)
            self.conv_layer_6 = make_convolution_layers(256, 256)
            self.pool_layer_4 = nn.MaxPool3d((temp1[1], temp2[1], temp3[1]), stride=2)

        if temp1[2] != 1 or temp2[2] != 1 or temp3[2] != 1:
            linear_number = 512
            self.conv_layer_7 = make_convolution_layers(256, 512)
            self.conv_layer_8 = make_convolution_layers(512, 512)
            self.pool_layer_5 = nn.MaxPool3d((temp1[2], temp2[2], temp3[2]), stride=2)

        self.avg_pool_layer = nn.Sequential(nn.AvgPool3d(kernel_size=1,stride=1))
        self.classifier = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(linear_number, 2),
        )

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.pool_layer_1(out)
        out = self.conv_layer_2(out)
        out = self.pool_layer_2(out)
        if self.temp1[0] != 1 or self.temp2[0] != 1 or self.temp3[0] != 1:
            out = self.conv_layer_3(out)
            # out = self.conv_layer_4(out)
            out = self.pool_layer_3(out)
        if self.temp1[1] != 1 or self.temp2[1] != 1 or self.temp3[1] != 1:
            out = self.conv_layer_5(out)
            # out = self.conv_layer_6(out)
            out = self.pool_layer_4(out)
        if self.temp1[2] != 1 or self.temp2[2] != 1 or self.temp3[2] != 1:
            out = self.conv_layer_7(out)
            out = self.conv_layer_8(out)
            out = self.pool_layer_5(out)
        # out = self.avg_pool_layer(out)
        out = out.view(out.size(0), -1)

        # intermediate_out = out
        out = self.classifier(out)
        return out
