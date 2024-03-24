import importlib
import os
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from utils.hparams import hparams, set_hparams
import numpy as np
from utils.utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars, load_ckpt, Measure
from data import Data
from option import args
from utils.zmerge import save_all
from torch.utils.data import dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_image(name,img,training_step,path):
    img=img.dot(255)
    Image.fromarray(np.uint8(img.transpose(1,2,0))).save(path+str(training_step)+'/'+name+".png")


class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.dataset_cls = None
        self.metric_keys = ['all_psnr', 'all_ssim' ,'all_lpips','all_lr_psnr','apsnr', 'assim', 'spsnr', 'sssim','upsnr', 'ussim']
        self.work_dir = hparams['work_dir']
        self.first_val = True
        self.loader = Data(args)
        self.measure = Measure(len(self.loader.loader_test))

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self):
        dataset = self.dataset_cls('train')  
        return torch.utils.data.DataLoader(
            dataset, batch_size=hparams['batch_size'], shuffle=True,
            pin_memory=False, num_workers=hparams['num_workers'])

    def build_val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls('valid'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls('test'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def train(self):
        model = self.build_model().cuda()
        optimizer = self.build_optimizer(model)
        #self.global_step = training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
        self.global_step = training_step=0
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        while(training_step<90000):  #all step
            for index,(batch,filename) in enumerate(self.loader.loader_train):
                if training_step % hparams['val_check_interval'] == 1:  # val
                    with torch.no_grad():
                        model.eval()
                        self.validate(training_step)
                    save_checkpoint(model, optimizer, self.work_dir, training_step, hparams['num_ckpt_keep'])
                model.train()
                optimizer.zero_grad()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.step()
                training_step += 1
                scheduler.step(training_step)
                self.global_step = training_step
                if training_step % 1000== 0:
                    print({f'tr/{k}': v for k, v in losses.items()}, training_step)


    def validate(self, training_step):
        val_dataloader = self.loader.loader_test
        metrics = {k: 0 for k in self.metric_keys}
        all_image=[]
        for index,(batch,filename) in enumerate(val_dataloader):
            if self.first_val and index > hparams['num_sanity_val_steps']:
                break
            if(index%10==0):
                print(index)
            if(index>10):
                break
            batch = move_to_cuda(batch)
            img, rrdb_out, ret = self.sample_and_test(batch,filename[0].split('_')[0], index)
            img_hr = batch['img_hr']
            img_lr = batch['img_lr']
            all_image.append(np.uint8(plot_img(img[0]).transpose(1,2,0)*255))
            if img is not None:
                if(not os.path.exists(('{}/image').format(self.work_dir))):
                    os.mkdir(r'{}/image'.format(self.work_dir))
                if(not os.path.exists('{}/image/{}'.format(self.work_dir,training_step))):
                    os.mkdir(r'{}/image/{}'.format(self.work_dir,training_step))
                if(hparams['sr_scale'] ==3):
                    if(index<10):
                        path=self.work_dir+"/image/"
                        save_image(f'{filename[0]}_HR',plot_img(img_hr[0]),training_step,path)
                        save_image(f'{filename[0]}_LR',plot_img(img_lr[0]),training_step,path)
                        save_image(f'{filename[0]}_SR',plot_img(img[0]),training_step,path)
                else:
                    if(filename[0] in ['argriculture_HR_1','argriculture_HR_10','argriculture_HR_11','argriculture_HR_12','special_HR_10','special_HR_102','special_HR_103','special_HR_104','urban_HR_10','urban_HR_100','urban_HR_101','urban_HR_102','urban_HR_103','urban_HR_104','urban_HR_105','urban_HR_106']):
                        path=self.work_dir+"/image/"
                        save_image(f'{filename[0]}_HR',plot_img(img_hr[0]),training_step,path)
                        save_image(f'{filename[0]}_LR',plot_img(img_lr[0]),training_step,path)
                        save_image(f'{filename[0]}_SR',plot_img(img[0]),training_step,path)
            for k in self.metric_keys:
                metrics[k]+=ret[k]
        if(not self.first_val and index>2000):
                save_all(self.work_dir+"/image/"+str(training_step)+'/all.tif',all_image, id=40)
        if hparams['infer']:
            print('Val results:', metrics)
        else:
            if not self.first_val:
                print('Val results:')
                print('fid:'+str(round(self.measure.all_fid(),5)))
                for k in self.metric_keys:
                    if(k in ['all_psnr','all_ssim','all_lpips']):
                        print(k+':'+str(round(metrics[k]/(index+1),5)))
            else:
                print('Sanity val results:', metrics)
        self.first_val = False

    def test(self):
        self.first_val=False
        model = self.build_model().cuda()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
        with torch.no_grad():
            model.eval()
            self.validate(1)
                            
    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == '__main__':
    set_hparams(config='./configs/diffsr_alsat4x.yaml',exp_name=args.save,hparams_str="rrdb_ckpt=checkpoints/rrdb_div2k_1") 
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    trainer.train()
