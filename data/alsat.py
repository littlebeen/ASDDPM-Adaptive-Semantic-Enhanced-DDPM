import os
import glob
from .srdata import SRData
from .common import np2Tensor
from .imgproc import imresize
import numpy as np

class ALSAT(SRData):
    def __init__(self, args,name='OLI2MSI', train=True, benchmark=False):
        super(ALSAT, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*' + '.jpg'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                filename = filename.replace('H','L')
                names_lr.append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, '.jpg'
                    )
                ))
            return names_hr, names_lr
        else:
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_test_hr, '*' + '.jpg'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_test_lr, '{}{}'.format(
                        filename, '.jpg'
                    )
                ))
            return names_hr, names_lr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  #whc
        lr,hr = self.get_patch(lr, hr)  #切片
        lr_up = imresize(lr.astype(np.float32), self.scale)  #把lr放大n倍
        #pair=lr,hr  #关闭切片
        lr,hr,lr_up = np2Tensor(*[lr,hr,lr_up], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        return {'img_lr':lr, 'img_hr':hr, 'img_lr_up': lr_up},filename  #cwh
    

    def _set_filesystem(self, dir_data):
        super(ALSAT, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Train/HR')
        self.dir_lr = os.path.join(self.apath, 'Train/LR')
        self.dir_test_hr = os.path.join(self.apath, 'Test/HR')
        self.dir_test_lr = os.path.join(self.apath, 'Test/LR')
        # self.dir_test_hr = os.path.join(self.apath, 'Test/Urban/HR')
        # self.dir_test_lr = os.path.join(self.apath, 'Test/Urban/LR')
        # self.dir_test_hr = os.path.join(self.apath, 'Test/Special/HR')
        # self.dir_test_lr = os.path.join(self.apath, 'Test/Special/LR')
        # self.dir_test_hr = os.path.join(self.apath, 'Test/Agriculture/HR')
        # self.dir_test_lr = os.path.join(self.apath, 'Test/Agriculture/LR')
