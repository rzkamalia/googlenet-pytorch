import torch
from torch import nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1x1, f3x3_reduce_1, f3x3_1, f3x3_reduce_2, f3x3_2, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.conv1x1 = nn.Conv2d(in_channels, f1x1, kernel_size = 1, stride = 1, padding = 0)
        self.batchnorm1x1 = nn.BatchNorm2d(f1x1)
        
        # 3x3 conv branch 1
        self.conv3x3_reduce_1 = nn.Conv2d(in_channels, f3x3_reduce_1, kernel_size = 1, stride = 1, padding = 0)
        self.batchnorm3x3_reduce_1 = nn.BatchNorm2d(f3x3_reduce_1)
        self.conv3x3_1 = nn.Conv2d(f3x3_reduce_1, f3x3_1, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm3x3_1 = nn.BatchNorm2d(f3x3_1)

        # 3x3 conv branch 2
        self.conv3x3_reduce_2 = nn.Conv2d(in_channels, f3x3_reduce_2, kernel_size = 1, stride = 1, padding = 0)
        self.batchnorm3x3_reduce_2 = nn.BatchNorm2d(f3x3_reduce_2)
        self.conv3x3_2 = nn.Conv2d(f3x3_reduce_2, f3x3_2, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm3x3_2 = nn.BatchNorm2d(f3x3_2)
        
        # max pooling branch
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size = 1, stride = 1, padding = 0)
        self.batchnorm_pool_proj = nn.BatchNorm2d(pool_proj)
    
    def forward(self, x):
        # 1x1 conv branch
        out_1x1 = self.conv1x1(x)
        out_1x1 = self.batchnorm1x1(out_1x1)
        
        # 3x3 conv branch 1
        out_3x3_reduce_1 = self.conv3x3_reduce_1(x)
        out_3x3_reduce_1 = self.batchnorm3x3_reduce_1(out_3x3_reduce_1)
        out_3x3_1 = self.conv3x3_1(out_3x3_reduce_1)
        out_3x3_1 = self.batchnorm3x3_1(out_3x3_1)
        
        # 3x3 conv branch 2
        out_3x3_reduce_2 = self.conv3x3_reduce_2(x)
        out_3x3_reduce_2 = self.batchnorm3x3_reduce_2(out_3x3_reduce_2)
        out_3x3_2 = self.conv3x3_2(out_3x3_reduce_2)
        out_3x3_2 = self.batchnorm3x3_2(out_3x3_2)
        
        # max pooling branch
        out_pool = self.pool(x)
        out_pool = self.pool_proj(out_pool)
        out_pool = self.batchnorm_pool_proj(out_pool)
        
        # concate the outputs of the branches along the channel dimension
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_pool], dim = 1)
        return out

class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        
        self.conv2a = nn.Conv2d(64, 64, kernel_size = 1, stride = 1, padding = 0)
        self.batchnorm2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 2)
        self.batchnorm2b = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        
        #  in_channels, f1x1, f3x3_reduce_1, f3x3_1, f3x3_reduce_2, f3x3_2, pool_proj
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.batchnorm2a(x)
        x = self.conv2b(x)
        x = self.batchnorm2b(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x