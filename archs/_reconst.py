
import torch
import torch.nn as nn

class PsrtUpsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if scale == 4:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        elif scale == 2:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(PsrtUpsample, self).__init__(*m)

class IartUpsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if scale == 4:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        elif scale == 2:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(IartUpsample, self).__init__(*m)
