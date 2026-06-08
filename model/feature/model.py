# torch libraries
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time

# customized libraries
from model.feature.create_backbone import Network, DimensionalReduction, Motion_guided_decoder
from model.feature.Res3dNet import MotionNet
from model.feature.Res3dNet import get_inplanes, Motion_Block

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class CoUpdater(nn.Module):
    def __init__(self, args=None):
        super(CoUpdater, self).__init__()

        self.args = args
        self.channel = args['channel']
        self.test_mode = args['test_mode']
        self.corr_levels = args['corr_levels']
        self.corr_radius = args['corr_radius']
        self.hidden_dim = args['hidden_dim']
        self.context_dim = args['context_dim']
        self.iters = args['iters']
        self.inp_size = args['inp_size']

        self.backbone = Network(channel=self.channel, pretrained=None, backbone_name=args['backbone_name'],
                                input_shape=args['in_channel_list'])
        # Motion net
        self.CNN_3D = MotionNet(Motion_Block, [1,1,1,1], get_inplanes())
        self.motion_guided_decoder = Motion_guided_decoder(self.channel)

        self.dr1_resnet50 = DimensionalReduction(64, self.channel)
        self.dr2_resnet50 = DimensionalReduction(128, self.channel)
        self.dr3_resnet50 = DimensionalReduction(320, self.channel)
        self.dr4_resnet50 = DimensionalReduction(512, self.channel)

        self.conv_motion = nn.Conv2d(512, 64, kernel_size=3, padding=1)

    def forward(self, images, nFrames):
        features = []
        for i in range(nFrames):
            features.append(self.backbone.feat_net(images[:,i,:,:,:]))
        # current frame feature
        current_feature = features[nFrames//2]
        f_1 = self.dr1_resnet50(current_feature[0])
        f_2 = self.dr2_resnet50(current_feature[1])
        f_3 = self.dr3_resnet50(current_feature[2])
        f_4 = self.dr4_resnet50(current_feature[3])

        # 3D motion modeling
        input_feature = [self.dr1_resnet50(x[0]) for x in features]
        input_feature = torch.stack(input_feature, 2)
        motion = self.CNN_3D(input_feature).squeeze(2)
        motion = self.conv_motion(motion)
        mask = self.motion_guided_decoder(f_4, f_3, f_2, f_1, motion)
        return mask


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size, input_size),  # (self.image_encoder.img_size, self.image_encoder.img_size)
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
