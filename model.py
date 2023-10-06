"""Some codes from https://github.com/alterzero/DBPN-Pytorch"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, activation='prelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.activation = activation
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class UpFusionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2):
        super(UpFusionBlock, self).__init__()
        self.up_conv1 = PSBlock(num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv3 = PSBlock(num_filter)

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpFusionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_groups=1):
        super(D_UpFusionBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_groups, num_filter, 1, 1, 0)
        self.up_conv1 = PSBlock(num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv3 = PSBlock(num_filter)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class DownFusionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2):
        super(DownFusionBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv2 = PSBlock(num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownFusionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_groups=1):
        super(D_DownFusionBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_groups, num_filter, 1, 1, 0)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv2 = PSBlock(num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0


class PSBlock(nn.Module):
    def __init__(self, n_feat):
        super(PSBlock, self).__init__()
        modules = []
        modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, activation=None))
        modules.append(nn.PixelShuffle(2))
        self.up = nn.Sequential(*modules)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.up(x)
        out = self.act(out)
        return out

class Net(nn.Module):    
    def __init__(self, opt):        

        output_size = opt.angular_out * opt.angular_out

        super(Net, self).__init__()

        # Initial Feature Extraction
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=output_size, kernel_size=(4,3,3), stride=(1,1,1), padding=(0,1,1))
        self.relu4 = nn.ReLU(inplace=True)

        # Channel Fusion Module
        kernel = 6
        stride = 2
        padding = 2

        num_groups = 12
        base_filter = output_size

        self.feat0 = ConvBlock(base_filter, base_filter, 3, 1, 1)
        
        self.up1 = UpFusionBlock(base_filter, kernel, stride, padding)
        self.down1 = DownFusionBlock(base_filter, kernel, stride, padding)
        self.up2 = UpFusionBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownFusionBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpFusionBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownFusionBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpFusionBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownFusionBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpFusionBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownFusionBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpFusionBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownFusionBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpFusionBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownFusionBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpFusionBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownFusionBlock(base_filter, kernel, stride, padding, 8)
        self.up9 = D_UpFusionBlock(base_filter, kernel, stride, padding, 8)
        self.down9 = D_DownFusionBlock(base_filter, kernel, stride, padding, 9)
        self.up10 = D_UpFusionBlock(base_filter, kernel, stride, padding, 9)
        self.down10 = D_DownFusionBlock(base_filter, kernel, stride, padding, 10)
        self.up11 = D_UpFusionBlock(base_filter, kernel, stride, padding, 10)
        self.down11 = D_DownFusionBlock(base_filter, kernel, stride, padding, 11)
        self.up12 = D_UpFusionBlock(base_filter, kernel, stride, padding, 11)
        self.down12 = D_DownFusionBlock(base_filter, kernel, stride, padding, 12)

        # LF Reconstruction
        self.output_conv = ConvBlock(num_groups * base_filter, output_size, 3, 1, 1, activation=None)

    def forward(self, inputs, opt):
        N,num_source,h,w = inputs.shape   #[N,num_source,h,w]
        inputs = inputs.view(N, 1, num_source, h, w)

        c1 = self.conv1(inputs)
        r1 = self.relu1(c1)

        c2 = self.conv2(r1)
        r2 = self.relu2(c2)

        c3 = self.conv3(r2)
        r3 = self.relu3(c3)

        c4 = self.conv4(r3)
        r4 = self.relu4(c4)

        N,c,d,h,w = r4.shape   #[N,c,d,h,w]
        out0 = r4.view(1,c,h,w)

        x = self.feat0(out0)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down8(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up9(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down9(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up10(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down10(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up11(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down11(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up12(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down12(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        x = self.output_conv(concat_l)

        return x
