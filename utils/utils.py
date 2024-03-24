import subprocess
import torch.distributed as dist
import glob
import os
import re
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .matlab_resize import imresize
import lpips
from .fid.inception import InceptionV3
from scipy import linalg
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from torch.nn import functional as F

loss_fn = lpips.LPIPS(net='alex', version=0.1)

def caculate_lpips(img0,img1):
    im1=np.copy(img0)
    im2=np.copy(img1)
    im1=torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance 


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors




def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_checkpoint(model, optimizer, work_dir):
    checkpoint, _ = get_last_checkpoint('./pre_train')
    # related_params={k:v for k,v in checkpoint['state_dict']['model'].items() if 'denoise_fn' in k}
    # related_params2={k:v for k,v in checkpoint['state_dict']['model'].items() if 'rrdb' in k}
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'], False)
        # model.load_state_dict(related_params, False)
        # model.load_state_dict(related_params2, False)
        model.cuda()
        print('load pretrain model from pre_train')
        #optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.cuda()
    return training_step


def save_checkpoint(model, optimizer, work_dir, global_step, num_ckpt_keep):
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def plot_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if '.' in k]) > 0:
            state_dict = {k[len(model_name) + 1:]: v for k, v in state_dict.items()
                          if k.startswith(f'{model_name}.')}
        else:
            state_dict = state_dict[model_name]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


def _compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    FID_val = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return FID_val

def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

class Measure:
    def __init__(self, leng, net='alex'):
        self.model = lpips.LPIPS(net=net)
        n_act = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[n_act]
        self.inception_model = InceptionV3([block_idx]).to('cuda')
        self.inception_model.eval()
        self.batch_size=1
        act1 = np.zeros((leng, n_act))
        act2 = np.zeros((leng, n_act))
        self.act = [act1, act2]

        self.inception_model2 = inception_v3(pretrained=True, transform_input=False).to('cuda')
        self.inception_model2.eval()
        self.preds = np.zeros((leng, 1000))
        self.length= leng
    
    def get_pred(self,x):
        # if resize:
        #     x = up(x)
        x = self.inception_model2(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def get_activations(self,x):
        x = self.inception_model(x)[0]
        return x.cpu().data.numpy().reshape(self.batch_size, -1)
    

    def caculate_fid(self,im1,im2,i):
        batch_size_i =1
        im1 = torch.from_numpy(im1.astype(np.float32)).to('cuda')
        im2 = torch.from_numpy(im2.astype(np.float32)).to('cuda')
        im1.unsqueeze_(0)
        im2.unsqueeze_(0)
        activation = self.get_activations(im1)
        self.act[0][i * self.batch_size:i * self.batch_size + batch_size_i] = activation
        activation2 = self.get_activations(im2)
        self.act[1][i * self.batch_size:i * self.batch_size + batch_size_i] = activation2
    
    def caculate_IS(self,im1,i):
        im1 = torch.from_numpy(im1.astype(np.float32)).to('cuda')
        im1.unsqueeze_(0)
        batch_size_i = 1
        self.preds[i * self.batch_size:i * self.batch_size + batch_size_i] = self.get_pred(im1)

    def all_fid(self):
        mu_act1, sigma_act1 = compute_act_mean_std(self.act[0])
        mu_act2, sigma_act2 = compute_act_mean_std(self.act[1])
        FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
        return FID
    
    def all_IS(self):
        split_scores = []
        splits=10
        for k in range(splits):
            part = self.preds[k * (self.length // splits): (k + 1) * (self.length // splits), :] # split the whole data into several parts
            py = np.mean(part, axis=0)  # marginal probability
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]  # conditional probability
                scores.append(entropy(pyx, py))  # compute divergence
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores)

    def measure(self, imgA, imgB, img_lr, sr_scale,index):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]  #up
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]  #hr
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        if isinstance(imgA, torch.Tensor):
            #imgA = np.round((imgA.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            #imgB = np.round((imgB.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            imgA = np.round(imgA.cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            imgB = np.round(imgB.cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            img_lr = np.round(img_lr.cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
        img_lr = img_lr.transpose(1, 2, 0)
        self.caculate_fid(imgA, imgB,index)
        #self.caculate_IS(imgA,index)
        lpips = caculate_lpips(imgA,imgB)
        imgA = imgA.transpose(1, 2, 0)
        imgB = imgB.transpose(1, 2, 0)
        imgA_lr = imresize(imgA, 1 / sr_scale)
        #imgA_lr =imgA
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lr_psnr = self.psnr(imgA_lr, img_lr)
        res = {'psnr': psnr, 'ssim': ssim,'lpips':lpips,'lpips': lpips, 'lr_psnr': lr_psnr}
        return {k: float(v) for k, v in res.items()}


    def ssim(self, imgA, imgB):
        score, diff = ssim(imgA, imgB, full=True, channel_axis=2, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1
