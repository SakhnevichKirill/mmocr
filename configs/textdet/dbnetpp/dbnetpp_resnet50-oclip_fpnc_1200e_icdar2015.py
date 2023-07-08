_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth'
# resume = True

# Set the maximum number of epochs to 400, and validate the model every 1 epochs
_base_.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))


batch_size = 8
num_workers = 16

_base_.train_dataloader.batch_size=batch_size
_base_.train_dataloader.num_workers = num_workers

_base_.auto_scale_lr.base_batch_size = batch_size
_base_.optim_wrapper.optimizer.lr = 0.001


_base_.val_dataloader.num_workers = num_workers
_base_.test_dataloader.num_workers = num_workers



param_scheduler = [
    dict(type='LinearLR', end=3, start_factor=0.001),
    dict(type='PolyLR', power=2, eta_min=1e-7, begin=3, end=40),
]


init_kwargs=dict(
    tags=["dbnetpp", "resnet50-oclip"], 
    notes="", 
)

_base_.visualizer.vis_backends[0].init_kwargs.update(init_kwargs)
