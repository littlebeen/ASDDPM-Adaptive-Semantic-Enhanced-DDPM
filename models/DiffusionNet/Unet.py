import functools
import torch
from torch import nn
import torch.nn.functional as F
from utils.hparams import hparams
from ..module_util import  initialize_weights
from ..util.commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from ..util.commons import ResnetBlock, Upsample, Block, Downsample,MeanShift

class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        if(hparams['sr_scale'] ==4):
            self.cond_proj = nn.ConvTranspose2d(cond_dim * ((hparams['rrdb_num_block'] + 1) // 3),
                                                dim, hparams['sr_scale'] * 2, hparams['sr_scale'],   #sr_scale 4
                                                hparams['sr_scale'] // 2) #反卷积
        else:
            self.cond_proj = nn.ConvTranspose2d(cond_dim * ((hparams['rrdb_num_block'] + 1) // 3),
                                                dim, 7, hparams['sr_scale'],
                                                2) #反卷积
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if hparams['use_attn']:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        if hparams['res'] and hparams['up_input']:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
            )
        if hparams['use_wn']:
            self.apply_weight_norm()
        if hparams['weight_init']:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up,noise=None):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        try:
            cond = self.cond_proj(torch.cat(cond[2::3], 1))
        except Exception as e:
             cond=0
        finally:
            pass
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond
                if hparams['res'] and hparams['up_input']:
                    x = x + self.up_proj(img_lr_up)  #不进去
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams['use_attn']:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        return x   #3 128 128

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

