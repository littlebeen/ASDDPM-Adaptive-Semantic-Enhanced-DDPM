# OLI2MSI inference and train
# with pretrain model in ./pretrain
python test.py --lr_encoder rrdb3 --diffusion_net unetdualfusion --data_train OLI2MSI --data_train_dir OLI2MSI --scale 3 --patch_size 96

python trainer.py --lr_encoder rrdb3 --diffusion_net unetdualfusion --data_train OLI2MSI --data_train_dir OLI2MSI --scale 3 --patch_size 96


# ALSAT inference and train
# with pretrain model in ./pretrain
python test.py --lr_encoder rrdb4 --diffusion_net unetdualfusion --data_train ALSAT --data_train_dir ALSAT --scale 4 --patch_size 128

python trainer.py --lr_encoder rrdb4 --diffusion_net unetdualfusion --data_train ALSAT --data_train_dir ALSAT --scale 4 --patch_size 128