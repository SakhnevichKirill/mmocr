_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth'

# Set the maximum number of epochs to 400, and validate the model every 1 epochs
_base_.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=1)

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))



batch_size = 8
num_workers = 16

_base_.train_dataloader.batch_size=batch_size
_base_.train_dataloader.num_workers = num_workers

_base_.auto_scale_lr.base_batch_size = batch_size
_base_.optim_wrapper.optimizer.lr = 0.003


_base_.val_dataloader.num_workers = num_workers
_base_.test_dataloader.num_workers = num_workers

param_scheduler = [
    dict(type='LinearLR', end=200, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=200, end=1200),
]
