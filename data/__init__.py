from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, dataset):
        super(MyConcatDataset, self).__init__(dataset)
        self.train = dataset.train

    def set_scale(self, idx_scale):
        self.dataset.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        if not args.test_only:
            module_name = args.data_train
            if module_name == 'OLI2MSI':
                from .oli2msi import OLI2MSI as D
            if module_name == 'ALSAT':
                from .alsat import ALSAT as D
            dataset=D(args)
            testset=D(args,train=False)
            self.loader_train = dataloader.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
            self.loader_test=dataloader.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
            )
