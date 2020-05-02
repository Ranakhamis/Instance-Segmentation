import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import List




class FPN(nn.Module):
    
    def __init__(self, in_channels = [2048, 1024, 512]):
        super().__init__()

        #Note that in_channels given in reverse order, as FPN starts from the last stage in ResNet101 (stage 5) .. 
        self.lateral_layers  = nn.ModuleList([
            nn.Conv2d(x, 256, kernel_size=1)
            for x in (in_channels)
        ])

        self.pred_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        #These extra layers here to add the tow additional stages (p6, p7) .. 
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
            for _ in range(2)
        ])
        

    def forward(self, convouts:List[torch.Tensor]):
       
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for _ in range(len(convouts)):
            out.append(x)


        j = len(convouts)
        for lat_layer in self.lateral_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=bilinear, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x
        
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        for downsample_layer in self.downsample_layers:
            out.append(downsample_layer(out[-1]))

        return out

