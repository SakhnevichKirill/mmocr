_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/custom_runtime.py',
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/schedules/schedule_adam_base.py',
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3 / (10**4) * 2048 / 2048,
        betas=(0.9, 0.99),
        eps=8e-8,
        weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        end_factor=1.0,
        end=6,
        verbose=False,
        convert_to_iter_based=True),
    # dict(
    #     # OneCi
    #     type='OneCycleLR',
    #     eta_max=0.01,
    #     div_factor=10,
    #     final_div_factor=20,
    #     pct_start=0.1,
    #     total_steps=100,
    #     verbose=False,
    #     by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=130,
        begin=20,
        end=150,
        verbose=False,
        convert_to_iter_based=True
    ),
]

# dataset settings
train_list = [
    # _base_.mjsynth_textrecog_train, 
    _base_.synthtext_textrecog_train, 
    _base_.icdar2015_textrecog_train
]

# test_list = [
#     _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test,
#     _base_.svt_textrecog_test, _base_.svtp_textrecog_test,
#     _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
# ]

test_list = [
    _base_.synthtext_textrecog_train, 
    _base_.synthtext_textrecog_test,
    _base_.icdar2015_textrecog_train,
    _base_.icdar2015_textrecog_test,
]


val_evaluator = dict(
    # dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15']
    dataset_prefixes=[
        'synthtext_train',
        'synthtext',
        'icdar2015_train',
        'icdar2015'
    ]
)

test_evaluator = val_evaluator

# # Use ConcatDataset to combine the datasets in the list
train_dataset = dict(
       type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
       type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=512,
    num_workers=16,
    # persistent_workers=True,
    # pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=128,
    num_workers=16,
    # persistent_workers=True,
    # pin_memory=True,
    # drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=512)


init_kwargs=dict(
    tags=["svtr", "tiny"], 
    notes="", 
)

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/custom_chars.txt',
    with_padding=True,
    with_unknown=True,
)

_base_.model.decoder.dictionary = dictionary

_base_.visualizer.vis_backends[0].init_kwargs.update(init_kwargs)

seed=0