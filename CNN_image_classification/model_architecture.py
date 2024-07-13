import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # nn.Conv2d(1, 32, 3, 1) means:
        # -- 1 input channel (i.e. a single image)
        # -- 32 output channels (i.e. 32 different convoluted images, each
        #                     from a different kernel. The convoluted
        #                     images are called feature maps.)
        # -- 3x3 kernel size
        # -- Stride of 1
        #
        # For an input of size 1 x L x W, the output size is
        # 32 x (L - 3 + 1) x (W - 3 + 1) due to the kernel size and stride.
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Randomly "drops out" 25% of channels from the preceding 
        # convolutionary layer. For example, if the previous layer 
        # has 32 output channels (i.e. 32 feature maps), then 25% 
        # percent of those feature maps will be set to zero before being
        # input into the next layer.
        self.dropout1 = nn.Dropout2d(0.25)

        # This dropout function is for the fully connected layers, and
        # it just zeros out 50% of the outputs.
        self.dropout2 = nn.Dropout(0.5)
        
        # fc stands for fully connected
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # I think the "forward" method is required anytime you make 
    # an nn.Module subclass...?
    def forward(self, x):
        """
        This neural network requires inputs of size 1x28x28. Here's why:
        -The first convulational layer outputs 32x26x26
        -The second convulational layer outputs 64x24x24
        -Max pooling reduces size to 64x12x12 = 9216
        -9216 is the size of the fully connected layer, so fc1 gets an
         input of the appropriate size.
        """

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # max_pool2d reduces the image heigh and width by half (number
        # of pixels decreases by factor of 4). The input is partitioned
        # into 2x2 regions, and the maximum-intensity pixel within
        # each region is retained.
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        # Image needs to be flattened before passing through fully
        # connected layers.
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
