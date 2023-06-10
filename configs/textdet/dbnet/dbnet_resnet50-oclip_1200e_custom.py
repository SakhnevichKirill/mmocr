_base_ = [
    'dbnet_resnet50-dcnv2_fpnc_100k_synthtext.py',
]


# Set the maximum number of epochs to 400, and validate the model every 1 epochs
_base_.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=1)

load_from = None

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.batch_size = 8
_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.002

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]
