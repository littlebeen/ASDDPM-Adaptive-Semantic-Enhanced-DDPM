import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch.utils.data as data


class DataProcess():
    def __init__(self):
        super(DataProcess, self).__init__()

    def lr_process(lr):
        return lr

    def sr_process(sr,lr): 
        return sr, lr #cwh


class SRData(data.Dataset):
    def __init__(self, args, name, train=True, benchmark=False):
        self.args = args
        self.name = args.data_train_dir
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'vdsr')
        self.scale = int(args.scale)
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            if train:  #训练集和测试集分开
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(args.scale)
                    ),
                    exist_ok=True
                )
                self.images_hr, self.images_lr = [], [] 
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    #b = b.replace('.TIF', '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(args.ext, h, b, verbose=True) 
                for i, l in enumerate(list_lr):
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
            else:
                os.makedirs(
                    self.dir_test_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(
                        self.dir_test_lr.replace(self.apath, path_bin),
                        'X{}'.format(args.scale)
                    ),
                    exist_ok=True
                )
                
                self.images_hr, self.images_lr = [], []
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    #b = b.replace('.TIF', '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(args.ext, h, b, verbose=True) 
                for i, l in enumerate(list_lr):
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                #self.repeat = max(n_patches // n_images, 1)  #repeat与batch_size有关
                self.repeat = 1
    # Below functions as used to prepare images
    def _scan(self):
        if(self.train):
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*' + '.png'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            return names_hr, names_lr
        else:
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_test_hr, '*' + '.png'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_test_lr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:  #切片，测试也得切片!!
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=False,
                input_large=False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else: #测试只取中间
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr