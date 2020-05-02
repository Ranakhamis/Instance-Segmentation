from ResidualBloock import *

class ResNet101(nn.Module):

    def __init__(self, layers = [3, 4, 23, 3], block=ResidualBlock):

        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.dilation = 1
        self.inputs = 64


        #These set of layers for satge zero .. 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #These functions calls for makeing stages [1, 2, 3, 4] ..
        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)


        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, outputs, blocks, stride=1):
        """ Helper function to creat set of layers for each stage """
        downsample = None
            
        downsample = nn.Sequential(
            nn.Conv2d(self.inputs, outputs * 4,
                        kernel_size=1, stride=stride, bias=False,
                        dilation=self.dilation),
            nn.BatchNorm2d(outputs * 4),
        )

        layers = []
        layers.append(block(self.inputs, outputs, stride, downsample, self.dilation))
        self.inputs = outputs * 4
        for i in range(1, blocks):
            layers.append(block(self.inputs, outputs))

        layer = nn.Sequential(*layers)

        self.channels.append(outputs * 4)
        self.layers.append(layer)

        return layer

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Helper function to load pre-trained weights for ResNet101 """
        state_dict = torch.load(path)

        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)


