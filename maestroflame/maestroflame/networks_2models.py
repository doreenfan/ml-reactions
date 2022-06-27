import torch.nn as nn
import torch
from .networks import *

class Combine2Models(nn.Module):
    def __init__(self, ModelSpec, ModelEnuc):
        super().__init__()
        self.ModelSpec = ModelSpec
        self.ModelEnuc = ModelEnuc

    def forward(self, x):
        xspec = self.ModelSpec(x)
        xenuc = self.ModelEnuc(x)
        return torch.cat((xspec, xenuc), dim=1)


class Net_2Models(nn.Module):
    def __init__(self, ModelSpec, input_size, h1, h2, h3, output_size, tanh=True):
        super().__init__()
        self.ModelSpec = ModelSpec
        self.fc1 = nn.Linear(input_size+2, h1)  # +2 is from the output species from ModelSpec()
        self.ac1 = nn.Tanh() if tanh else nn.CELU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh() if tanh else nn.CELU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh() if tanh else nn.CELU()
        self.fc4 = nn.Linear(h3, output_size-2)  # -2 is also from output species

    def forward(self, x):
        xspec = self.ModelSpec(x)
        x = torch.cat((x, xspec), dim=1)
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.ac3(self.fc3(x))
        x = self.fc4(x)
        x = torch.cat((xspec, x), dim=1)
        return x

# Net inspired by U-Net
class U_Net_2Models(nn.Module):
    def __init__(self, ModelSpec, input_size, h1, h2, h3, h4, output_size, tanh=True):
        super().__init__()
        self.ModelSpec = ModelSpec
        self.fc1 = nn.Linear(input_size+2, h1)  # +2 is from the output species from ModelSpec()
        self.ac1 = nn.Tanh() if tanh else nn.CELU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh() if tanh else nn.CELU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh() if tanh else nn.CELU()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh() if tanh else nn.CELU()
        self.fc5 = nn.Linear(h4, output_size-2)  # -2 is also from output species

        # layers between non-consecutive layers
        self.io = nn.Linear(input_size+2, output_size-2)
        self.fc1to4 = nn.Linear(h1, h4)

    def forward(self, x):
        xspec = self.ModelSpec(x)
        x = torch.cat((x, xspec), dim=1)
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2))
        x4 = self.fc4(x3)
        x4 = self.ac4(x4 + self.fc1to4(x1))
        x5 = self.fc5(x4) + self.io(x)
        x6 = torch.cat((xspec, x5), dim=1)
        return x6

# Nets inspired by ResNet
class ResNet_2Models(nn.Module):
    def __init__(self, ModelSpec, input_size, h1, h2, h3, h4, output_size, tanh=True):
        super().__init__()
        self.ModelSpec = ModelSpec
        self.fc1 = nn.Linear(input_size+2, h1)  # +2 is from the output species from ModelSpec()
        self.ac1 = nn.Tanh() if tanh else nn.CELU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh() if tanh else nn.CELU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh() if tanh else nn.CELU()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh() if tanh else nn.CELU()
        self.fc5 = nn.Linear(h4, output_size-2)  # -2 is also from output species
        
        # layers between non-consecutive layers 
        self.fc0to3 = nn.Linear(input_size+2, h3)
        self.fc2to5 = nn.Linear(h2, output_size-2)

    def forward(self, x):
        xspec = self.ModelSpec(x)
        x = torch.cat((x, xspec), dim=1)
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2) + self.fc0to3(x))
        x4 = self.ac4(self.fc4(x3))
        x5 = self.fc5(x4) + self.fc2to5(x2)
        x6 = torch.cat((xspec, x5), dim=1)
        return x6

class Cross_ResNet_2Models(nn.Module):
    def __init__(self, ModelSpec, input_size, h1, h2, h3, h4, output_size, tanh=True):
        super().__init__()
        self.ModelSpec = ModelSpec
        self.fc1 = nn.Linear(input_size+2, h1)  # +2 is from the output species from ModelSpec()
        self.ac1 = nn.Tanh() if tanh else nn.CELU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh() if tanh else nn.CELU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh() if tanh else nn.CELU()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh() if tanh else nn.CELU()
        self.fc5 = nn.Linear(h4, output_size-2)  # -2 is also from output species
        
        # layers between non-consecutive layers
        self.fc1to4 = nn.Linear(h1, h4)
        self.fc2to5 = nn.Linear(h2, output_size-2)

    def forward(self, x):
        xspec = self.ModelSpec(x)
        x = torch.cat((x, xspec), dim=1)
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2))
        x4 = self.ac4(self.fc4(x3) + self.fc1to4(x1))
        x5 = self.fc5(x4) + self.fc2to5(x2)
        x6 = torch.cat((xspec, x5), dim=1)
        return x6

