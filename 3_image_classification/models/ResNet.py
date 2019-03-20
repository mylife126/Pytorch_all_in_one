import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBlock(nn.Module):
    
    """Resnet block for resnet18 and resnet 34
    
    
    """
    # the expand coefficient after this block 
    expand = 1    
    
    def __init__(self, in_channel, out_channel, stride):
        super(SimpleBlock, self).__init__()
        
        # forward without shortcut
        self.plain = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel)
                )
        
        # forward with shortcut
        # use option 2 in the paper, applying 1x1 convolution
        # if stride is not 1, then this block is in the begining, and apply conv shortcut
        if stride != 1:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel)
                    )
        # if stride is 1, then this block is in the middle, apply identity shortcut
        # use blank nn.Sequential() to simulate identity
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
      
        out = self.plain(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckBlock(nn.Module):
    """Resnet block for resnet50, 101 and 152
    
    """       
    expand = 4
    
    def __init__(self, in_channel, out_channel, stride):
        super(BottleneckBlock, self).__init__()
        
        # forward without shortcut
        self.plain = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel * self.expand, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel * self.expand)
                )
        # forward with shortcut
        # use option 2 in the paper, applying 1x1 convolution
        # if stride is not 1, then this block is in the begining, and apply conv shortcut
        # for the conv2, we want the stride equals to 1, so we should specifiy another condition when in_channel 
        # is not equal to out_channel* self.expand
        if stride != 1 or in_channel != out_channel * self.expand:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel * self.expand, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel * self.expand)
                    )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self, x):
        out = self.plain(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    """Designed for input size of 32, 3 channels image
    
    The main idea is the keep the architecure after size number 28, but modify the architecture before that part
    
    """
    
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        
        # keep track of current in_channel number
        self.in_channel = 64
        
        # conv1
        # (batch, 3, 32, 32) -> (batch, 64, 32, 32)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()                      
                )

        # conv2_x
        # Simple: (batch, 64, 32, 32) -> (first block)(batch, 64, 32, 32) -> (other blocks)(batch, 64, 32, 32)
        # Bottle: (batch, 64, 32, 32) -> (first block)(batch, 64, 32, 32) -> (middle blocks)(batch, )
        self.conv2 = self.make_layer(block, num_blocks[0], 1, 64)
        
        # conv3_x
        # (batch, 64, 32, 32) -> (first block)(batch, 128, 16, 16) -> (other blocks)(batch, 128, 16, 16)
        self.conv3 = self.make_layer(block, num_blocks[1], 2, 128)
        
        # conv4_x
        # (batch, 128, 16, 16) -> (first block)(batch, 256, 8, 8) -> (other blocks)(batch, 256, 8, 8)
        self.conv4 = self.make_layer(block, num_blocks[2], 2, 256)

        # conv5_x
        # (batch, 256, 8, 8) -> (first block)(batch, 512, 4, 4) -> (other blocks)(batch, 512, 4, 4)
        self.conv5 = self.make_layer(block, num_blocks[3], 2, 512)  
        
        # average pooling 
        # (batch, 512, 4, 4) -> (batch, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # fc layer
        # (batch, 512, 1, 1) - > (batch, 512) -> (batch, num_classes)
        self.fc = nn.Linear(512 * block.expand, num_classes)
        
    def make_layer(self, block, num_blocks, stride, out_channel):
        
        layers = []
        for n in range(num_blocks):
            # the first block has stride # stride. In channel and out channel may not be the same
            # others have stride # 1, in channel and out channel are the same
#            print(n)
            if n == 0:
                layers.append(block(self.in_channel, out_channel, stride))
            else:
                layers.append(block(self.in_channel, out_channel, 1))
            # update in_channel
            self.in_channel = out_channel * block.expand
            
        return nn.Sequential(*layers)
            
    def forward(self, x):
        out = self.conv1(x)
#        print("conv1")
#        print(out.shape)
        out = self.conv2(out)
#        print("conv2")
#        print(out.shape)
        out = self.conv3(out)
#        print("conv3")
#        print(out.shape)
        out = self.conv4(out) 
#        print("conv4")
#        print(out.shape)
        out = self.conv5(out)  
#        print("conv5")
#        print(out.shape)
        out = self.avgpool(out)    
        out = out.view(out.size(0), -1)
        embeddings = out
        out = self.fc(out)       
        return out, embeddings
        
        
def ResNet18(num_classes=5):
    return ResNet(SimpleBlock, [2, 2, 2, 2], num_classes)        
              
def ResNet34(num_classes=5):
    return ResNet(SimpleBlock, [3, 4, 6, 3], num_classes)        
            
def ResNet50(num_classes=5):
    return ResNet(BottleneckBlock, [3,4,6,3], num_classes)

def ResNet101(num_classes=5):
    return ResNet(BottleneckBlock, [3,4,23,3], num_classes)

def ResNet152(num_classes=5):
    return ResNet(BottleneckBlock, [3,8,36,3], num_classes)
              
        
        