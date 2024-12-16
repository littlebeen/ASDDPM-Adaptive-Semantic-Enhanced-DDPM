import os.path

import torch
from models.DiffusionNet.Unet import Unet
from models.DiffusionNet.DiT import DiT
from models.DiffusionNet.Unetdual import UnetDual
from models.DiffusionNet.Unetdualfusion import UnetDualFusion
from models.LREncoder.RRDB.rrdb4 import RRDBNet4
from models.LREncoder.RRDB.rrdb3 import RRDBNet3
from trainer import Trainer
from utils.hparams import hparams
from data.alsat import ALSAT

class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        print('diffusion_net',hparams['diffusion_net'])
        if(hparams['diffusion_net']=='unet'):
            denoise_fn = Unet(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if(hparams['diffusion_net']=='unetdual'):
            denoise_fn = UnetDual(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if(hparams['diffusion_net']=='unetdualfusion'):
            denoise_fn = UnetDualFusion(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if(hparams['diffusion_net']=='dit'):
            denoise_fn = DiT(dim=hidden_size,input_size=hparams['patch_size'])
        print('Total params: %.2fM' % (sum(p.numel() for p in denoise_fn.parameters())/1000000.0))
        if hparams['use_rrdb']:
            print('lr_encoder:',hparams['lr_encoder'])
            if(hparams['lr_encoder']=='rrdb4'):
                rrdb = RRDBNet4(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                          hparams['rrdb_num_feat'] // 2)
                print('rrdb4 is load')
                rrdb.load_state_dict(torch.load('./models/LREncoder/pretrain/model_best4.pt'),strict=False)
            if(hparams['lr_encoder']=='rrdb4v'):
                rrdb = RRDBNet4(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                          hparams['rrdb_num_feat'] // 2)
                print('rrdb4v is load')
                rrdb.load_state_dict(torch.load('./models/LREncoder/pretrain/model_best4v.pt'),strict=False)
            if(hparams['lr_encoder']=='rrdb3'):
                rrdb = RRDBNet3(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                          hparams['rrdb_num_feat'] // 2)
                print('rrdb3 is load')
                rrdb.load_state_dict(torch.load('./models/LREncoder/pretrain/model_best3.pt'),strict=False)
            if(hparams['lr_encoder']=='rrdb4p'):
                rrdb = RRDBNet4(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                          hparams['rrdb_num_feat'] // 2)
                print('rrdb4p is load')
                rrdb.load_state_dict(torch.load('./models/LREncoder/pretrain/model_best4p.pt'),strict=False)

            print('Lr encoder total params: %.2fM' % (sum(p.numel() for p in rrdb.parameters())/1000000.0))
        else:
            rrdb = None
        if(hparams['diff_type']=='diff'):
            from models.diffusion import GaussianDiffusion as M
            print('diff_type: diff')
        self.model = M(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=hparams['timesteps'],   #100
        )
        self.global_step = 0
        return self.model

    def sample_and_test(self, sample,filename,index):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_lr_up = sample['img_lr_up']
        img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape)
        for b in range(img_lr_up.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'],index)
            ret['all_psnr'] += s['psnr']
            ret['all_ssim'] += s['ssim']
            ret['all_lpips'] += s['lpips']
            ret['all_lr_psnr'] += s['lr_psnr']
            ret['n_samples'] += 1
            if(filename=='argriculture'):
                ret['apsnr']+=s['psnr']
                ret['assim']+=s['ssim']
            if(filename=='urban'):
                ret['upsnr']+=s['psnr']
                ret['ussim']+=s['ssim']
            if(filename=='special'):
                ret['spsnr']+=s['psnr']
                ret['sssim']+=s['ssim']
        return img_sr, rrdb_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_rrdb']:
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss


class SRDiffDf2k(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ALSAT