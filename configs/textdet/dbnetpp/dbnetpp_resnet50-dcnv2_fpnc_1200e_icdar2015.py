_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    # '../_base_/default_runtime.py',
    '../_base_/custom_runtime.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth'  # noqa

# # dataset settings
# train_list = [_base_.icdar2015_textdet_train]
# test_list = [_base_.icdar2015_textdet_test]

# # List of training datasets
train_list = [_base_.icdar2015_textdet_train, _base_.synthtext_textdet_train] 
# List of testing datasets
val_list = [
    _base_.icdar2015_textdet_test,
    _base_.synthtext_textdet_test
]


val_evaluator = dict(
    dataset_prefixes=['icdar2015', 'synthtext']
)
test_evaluator = val_evaluator

# # Use ConcatDataset to combine the datasets in the list
train_dataset = dict(
       type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
val_dataset = dict(
       type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)

# TODO: resize
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

# test_evaluator = dict(
#     dataset_prefixes=['icdar2015']
# )

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=8,
#     persistent_workers=True,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#        type='ConcatDataset', datasets=[_base_.icdar2015_textdet_test,], pipeline=_base_.test_pipeline))



auto_scale_lr = dict(base_batch_size=8)
