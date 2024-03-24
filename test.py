import importlib

from utils.hparams import hparams, set_hparams

from option import args


if __name__ == '__main__':
    set_hparams(config='./configs/diffsr_alsat4x.yaml',exp_name=args.save,hparams_str="rrdb_ckpt=checkpoints/rrdb_div2k_1") 
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    trainer.test()
