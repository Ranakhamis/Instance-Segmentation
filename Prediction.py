import torch
import torch.nn as nn
from itertools import product
from math import sqrt



class PredictionModule(nn.Module):
   
    
    def __init__(self, in_channels, out_channels=256, aspect_ratios=[[1]], scales=[1], clone=None):
        super().__init__()
      
        self.num_classes = 81
        self.prototypes    = 32
        self.anchors  = 3
        self.clone      = [clone]
      
        if clone is None:
            self.upfeature = nn.Sequential(nn.Conv2d(256, 256,kernel_size = 3,padding = 1),nn.ReLU(inplace=True))
            self.bbox_layer = nn.Conv2d(out_channels, self.anchors * 4, kernel_size = 3, padding = 1)
            self.conf_layer = nn.Conv2d(out_channels, self.anchors * self.num_classes, kernel_size = 3, padding = 1)
            self.mask_layer = nn.Conv2d(out_channels, self.anchors * self.prototypes, kernel_size = 3, padding = 1)
            
        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None



    def forward(self, x):
        
        src = self if self.clone[0] is None else self.clone[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)

        x = src.upfeature(x)
        bbox = src.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = src.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.prototypes)
        mask = torch.tanh(mask)
                
        
        priors = self.make_bbox(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
        
        return preds
    
    def make_bbox(self, conv_h, conv_w):
        """ Note that bbox are [x,y,width,height] where (x,y) is the center of the box. """
        prior_data = []
        for j, i in product(range(conv_h), range(conv_w)):
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h
            
            for scale, ars in zip(self.scales, self.aspect_ratios):
                for ar in ars:
                    
                    ar = sqrt(ar)
                    w = scale * ar / 550
                    h = scale / ar / 550

                    prior_data += [x, y, w, h]
        self.priors = torch.Tensor(prior_data).view(-1, 4)
        return self.priors





