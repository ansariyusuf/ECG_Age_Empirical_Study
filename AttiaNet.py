#Imports
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchinfo import summary
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import random
import numpy as np

"""Network implementation as defined by Attia et al. in Age and Sex Estimation Using Artificial
Intelligence From Standard 12-Lead ECGs (2019)"""

#define network modules
class Temporal_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, pooling_factor):
        super(Temporal_Block, self).__init__()
        
        #temporal convolution
        self.temporal_conv = nn.Conv1d(in_channels, num_filters, kernel_size, stride=1, padding='same')
        #Batchnorm post-convolution
        self.batchnorm = nn.BatchNorm1d(num_filters)
        #pooling to reduce temporal resolution
        self.pool = nn.MaxPool1d(pooling_factor, stride=pooling_factor)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = F.relu(self.batchnorm(x))
        x = self.pool(x)
        
        return x

class Spatial_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters, pooling_factor):
        super(Spatial_Block, self).__init__()
        
        #temporal convolution
        self.temporal_conv = nn.Conv1d(in_channels, num_filters, kernel_size, stride=1, padding='same')
        #Batchnorm post-convolution
        self.batchnorm = nn.BatchNorm1d(num_filters)
        #pooling to reduce temporal resolution
        self.pool = nn.MaxPool1d(pooling_factor, stride=pooling_factor)

    def forward(self, x):
        #make the spatial dimension the convolution dimension
        x = torch.permute(x, (0, 2, 1))
        x = self.temporal_conv(x)
        x = F.relu(self.batchnorm(x))
        x = self.pool(x)
        #restore the shape to orignal as intended
        x = torch.permute(x, (0, 2, 1))
        return x
    

class Fully_Connected(nn.Module):
    def __init__(self, inchannel, num_neurons, dropout_rate):
        super(Fully_Connected, self).__init__()
        
        self.fc = nn.Linear(inchannel, num_neurons)
        self.batchnorm = nn.BatchNorm1d(num_neurons)
        self.drop = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.batchnorm(x))
        x = self.drop(x)
        
        return x
    
class Attia_Net(nn.Module):
    def __init__(self,num_leads, inp_size):
        super(Attia_Net, self).__init__()
        
        #define hyperparemeters for layers
        self.kernel_sizes = [7,5,5,5,5,3,3,3]
        self.num_filters_perblock = [16,16,32,32,64,64,64,64]
        self.pooling_factors = [2,4,2,4,2,2,2,2]
        
        #define temporal layers
        self.temp1 = Temporal_Block(num_leads, self.kernel_sizes[0], self.num_filters_perblock[0], self.pooling_factors[0])
        self.temp2 = Temporal_Block(self.num_filters_perblock[0], self.kernel_sizes[1], self.num_filters_perblock[1], self.pooling_factors[1])
        self.temp3 = Temporal_Block(self.num_filters_perblock[1], self.kernel_sizes[2], self.num_filters_perblock[2], self.pooling_factors[2])
        self.temp4 = Temporal_Block(self.num_filters_perblock[2], self.kernel_sizes[3], self.num_filters_perblock[3], self.pooling_factors[3])
        self.temp5 = Temporal_Block(self.num_filters_perblock[3], self.kernel_sizes[4], self.num_filters_perblock[4], self.pooling_factors[4])
        self.temp6 = Temporal_Block(self.num_filters_perblock[4], self.kernel_sizes[5], self.num_filters_perblock[5], self.pooling_factors[5])
        self.temp7 = Temporal_Block(self.num_filters_perblock[5], self.kernel_sizes[6], self.num_filters_perblock[6], self.pooling_factors[6])
        self.temp8 = Temporal_Block(self.num_filters_perblock[6], self.kernel_sizes[7], self.num_filters_perblock[7], self.pooling_factors[7])
        
        #Add spatial block
        self.spatial = Spatial_Block(inp_size//1024, 12, 128, 2)
        #define fully connected layers
        #Note: input to FC is 4096 for input size 5120 
        self.FC1 = Fully_Connected(4096, 128, 0.5) 
        self.FC2 = Fully_Connected(128, 64, 0.5)
        
        #define output layer
        self.output = nn.Linear(64, 1)
    
    
    def forward(self, x):
        #print("feature map size:", x.size())
        x = self.temp1(x)
        #print("feature map size:", x.size())
        x = self.temp2(x)
        #print("feature map size:", x.size())
        x = self.temp3(x)
        #print("feature map size:", x.size())
        x = self.temp4(x)
        #print("feature map size:", x.size())
        x = self.temp5(x)
        #print("feature map size:", x.size())
        x = self.temp6(x)
        #print("feature map size:", x.size())
        x = self.temp7(x)
        #print("feature map size:", x.size())
        x = self.temp8(x)
        #print("feature map size:", x.size())

        x = self.spatial(x)
        #print("feature map size post spatial:", x.size())
        
        
        #Flatten output of spatial convolution
        x = x.reshape(x.size(0),-1)
        
        #pass through FC layers
        x = self.FC1(x)
        x = self.FC2(x)
        
        #generate the output
        return self.output(x)

Attia_network = Attia_Net(num_leads=12, inp_size = 1024)

#Attia_network = Attia_Net_100()
#test the newtork
random_data = torch.randn(20, 1, 1024)
print(Attia_network)