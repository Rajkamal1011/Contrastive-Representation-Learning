import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.in_channels = 3
        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #Changed first parameter from 512 to 256
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*2*2, 1000)  
        
    def forward(self, x):
        # layer_outputs = []
        for layer in self.conv_layers:
            x = layer(x)
            # print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        return x
            
        
# =512*7*7
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes = 10):
        super(Classifier, self).__init__()
        self.num_classes = 10
        self.feature_dim = feature_dim
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=self.num_classes)
        )
        
    def forward(self, x):
        # print("WUBBALUBBADUBDUB",self.feature_dim)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        # print(x.shape)
        
        return x

# Instantiate Encoder
# encoder = Encoder()

# # Generate random input tensor of size 32x32 with 3 channels (batch size = 1)
# input_tensor = torch.randn(1, 3, 32, 32)

# # Pass input through encoder
# output_tensor = encoder(input_tensor)

# # Print the shapes of output tensors from convolutional layers
# for i, layer_output in enumerate(output_tensor):
#     print(f"Layer {i+1} output shape: {layer_output.shape}")