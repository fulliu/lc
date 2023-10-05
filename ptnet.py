import torch
import logging
from operator import itemgetter
from torch import nn, Tensor
from model import cdpn_resnet, zebra_DeepLabV3

logger = logging.getLogger(__name__)

class ptnet(nn.Module):
    def __init__(self,
        cfg,
        cfg_global,
        total_bit_cnt = 0,
        ) -> None:
        super().__init__()
        net_name = cfg.net_name

        binary_bits = total_bit_cnt

        channel_dict = {}
        sparse_cnt = cfg_global.get('sparse_cnt',0)
        with_sparse = sparse_cnt > 3

        logger.info(f'Model:: total_bit_cnt: {total_bit_cnt}')

        if with_sparse:
            channel_dict['kpt_logits'] = sparse_cnt
        else:
            noc_key = 'xyz_noc_bin' if binary_bits > 0 else 'xyz_noc'
            channel_dict[noc_key] = max(1, binary_bits) if binary_bits>0 else 3
            self.noc_key = noc_key
            channel_dict['xyz_weights'] = 2
            channel_dict['msk_vis'] = 1
        
        channel_slices, start_channel = {}, 0
        for k,v in channel_dict.items():
            channel_slices[k]=slice(start_channel, start_channel + v)
            start_channel += v

        in_channels, dense_out_channels, kwargs = 3, sum(channel_dict.values()), cfg.net_config
        self.net = eval(f'{net_name}.get_network(in_channels, dense_out_channels, **kwargs)')

        if not with_sparse:
            self.weight_scale_layer = nn.Linear(self.net.feature_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

        self.binary_bits = binary_bits
        self.channel_slices = channel_slices
        self.cfg = cfg

    def forward(self, rgb):
        out_raw, feature = self.net(rgb)
        out_splitted = {k:out_raw[:,v] for k,v in self.channel_slices.items()}
        out_dict = {}

        if 'kpt_logits' in out_splitted:
            kpt_logits = out_splitted['kpt_logits']
            pts2d, pts2d_std, = softargmax_2d_std(kpt_logits.flatten(start_dim=-2).softmax(dim=-1).reshape_as(kpt_logits))
            out_dict.update({
                'pts2d':pts2d,
                'pts2d_std':pts2d_std
            })
            return out_dict

        noc_key = self.noc_key
        noc, xyz_weight_logits = itemgetter(noc_key, 'xyz_weights') (out_splitted)

        xyz_weight_scale:Tensor = self.weight_scale_layer(feature.flatten(start_dim=-2, end_dim=-1).mean(dim=-1)).exp()
        xyz_weight_scale = xyz_weight_scale[...,None,None]

        out_dict.update({
            noc_key: noc,
            'xyz_weight_logits':xyz_weight_logits,
            'xyz_weights_scale':xyz_weight_scale,
        })

        out_dict['msk_vis_logits'] = out_splitted['msk_vis']

        return out_dict


def softargmax_1d_cov(prob1d:Tensor):
    '''
    prob1d: (*, N)
    return:
    mean: (*,)
    cov:  (*,)
    '''
    prob1d = prob1d[...,None]   #(*, N, 1)
    xx = torch.arange(0, prob1d.shape[-2] - 0.5, device=prob1d.device, dtype = prob1d.dtype).unsqueeze(-2)    #(1, N)
    mx = xx @ prob1d  # (1, N)@(*, N, 1)=(*, 1, 1)
    dx = xx - mx #(1, N) - (*, 1, 1) = (*, 1, N)
    covx = dx ** 2 @ prob1d     #(*, 1, N) @ (*, N, 1) = (*, 1, 1)
    return mx[...,0,0], covx[...,0,0]   #(*,) (*,)


def softargmax_2d_std(prob2d:torch.Tensor, clamp_std = False):
    '''
    prob2d: (*, H, W)
    return:
    mean: (*, 2)
    cov:  (*, 2)
    '''
    mx, cx = softargmax_1d_cov(prob2d.sum(dim=-2))   #(*,),(*,)
    my, cy = softargmax_1d_cov(prob2d.sum(dim=-1))   #(*,),(*,)
    mean = torch.stack((mx,my),dim=-1)
    cov = torch.stack((cx,cy),dim=-1)
    std = (cov + 1e-6).sqrt()
    if clamp_std:
        small_std = std < 1
        std = torch.where(small_std, torch.exp((std-1)*small_std), std)
    return mean, std, 

