import torch
import torch.nn as nn

class _DenseLayer(nn.Sequential):
    """Origninated from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features)),
        self.add_module('elu', nn.ELU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_input_features, growth_rate, 
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        # Concatenation 
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
    
class _DenseBlock(nn.Sequential):
    """Origninated from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, 
                                growth_rate=growth_rate, 
                                bn_size=bn_size, 
                                drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
    
class Generator(nn.Module):
    """Origninated from Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        ngpu (int) - how many GPU you use.
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(self, growth_rate=16, block_config=(4, 4, 4, 4),
                 bn_size=2, drop_rate=0):

        super(Generator, self).__init__()
        # First convolution
        self.conv0 = nn.Conv2d(1, 2*growth_rate, kernel_size=3, padding=1, bias=False)

        # Each denseblock
        num_features = 2 * growth_rate  #2k
        num_features_cat = num_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[0]* growth_rate + num_features
        self.comp0 = nn.Conv2d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)
        
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[1]* growth_rate + num_features
        self.comp1 = nn.Conv2d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)        
        
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[2]* growth_rate + num_features
        #self.comp2 = nn.Conv2d(num_features_cat, num_features,
        #                       kernel_size=1, stride=1, bias=False)   
  
        #self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features,
        #                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #num_features_cat += block_config[3]* growth_rate + num_features
        self.recon = nn.Conv2d(num_features_cat, 1,
                               kernel_size=1, stride=1, bias=False)      

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv0(x)
        out = self.block0(x)
        features = torch.cat([x,out],1)
        out = self.comp0(features)
        
        out = self.block1(out)
        features = torch.cat([features,out],1)
        out = self.comp1(features)
    
        out = self.block2(out)
        features = torch.cat([features,out],1)
        #out = self.comp2(features)
    
        #out = self.block3(out)
        #features = torch.cat([features,out],1)
        out = self.recon(features)
        return out
                
class Discriminator(nn.Module):
    """Origninated from SRGAN paper, see `"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" <https://arxiv.org/abs/1609.04802>`_ 
    Args:
        ngpu (int) - how many GPU you use.
        cube_size (int) - the size of one patch (eg. 64 means a cubic patch with size: 64x64x64), this is exact the size of the model input.
    """
    def __init__(self, cube_size=256):
        super(Discriminator, self).__init__()
        num_features = 1
        self.main = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(1),

            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([num_features,cube_size//2,cube_size//2]),
            nn.LeakyReLU(1),

            nn.Conv2d(num_features, 2*num_features, kernel_size=3, padding=1),
            #nn.LayerNorm([2*num_features,cube_size//2,cube_size//2]),
            nn.LayerNorm([2*num_features,cube_size//2,cube_size//2]),
            nn.LeakyReLU(1),

            nn.Conv2d(2*num_features, 2*num_features, kernel_size=3, stride=2, padding=1),
            #nn.LayerNorm([2*num_features,cube_size//4,cube_size//4]),
            nn.LayerNorm([2*num_features,cube_size//4,cube_size//4]),
            nn.LeakyReLU(1),

            nn.Conv2d(2*num_features, 4*num_features, kernel_size=3, padding=1),
            #nn.LayerNorm([4*num_features,cube_size//4,cube_size//4]),
            nn.LayerNorm([4*num_features,cube_size//4,cube_size//4]),
            nn.LeakyReLU(1),

            nn.Conv2d(4*num_features, 4*num_features, kernel_size=3, stride=2, padding=1),
            #nn.LayerNorm([4*num_features,cube_size//8,cube_size//8]),
            nn.LayerNorm([4*num_features,cube_size//8,cube_size//8]),
            nn.LeakyReLU(1),

            nn.Conv2d(4*num_features, 8*num_features, kernel_size=3, padding=1),
            #nn.LayerNorm([8*num_features,cube_size//8,cube_size//8]),
            nn.LayerNorm([8*num_features,cube_size//8,cube_size//8]),
            nn.LeakyReLU(1),

            nn.Conv2d(8*num_features, 8*num_features, kernel_size=3, stride=2, padding=1),
            #nn.LayerNorm([8*num_features,cube_size//16,cube_size//16]),
            nn.LayerNorm([8*num_features,cube_size//16,cube_size//16]),
            nn.LeakyReLU(1),

            # different from the original SRGAN, we replaced the FC layers by global averaging pooling and convolution layers.
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8*num_features, 16*num_features, kernel_size=1),
            nn.LeakyReLU(1),
            nn.Conv2d(16*num_features, 1, kernel_size=1)
        )

        
    def forward(self, x):
        out = self.main(x)
        return out.view(out.size()[0])                

