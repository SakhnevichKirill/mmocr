_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/custom_runtime.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/icdar2015.py',
    # '../_base_/schedules/schedule_sgd_100k.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
# train_dataset = _base_.synthtext_textdet_train
# train_dataset.pipeline = _base_.train_pipeline
# test_dataset = _base_.icdar2015_textdet_test
# test_dataset.pipeline = _base_.test_pipeline

# # List of training datasets
train_list = [_base_.synthtext_textdet_train] 
# List of testing datasets
test_list = [
    _base_.icdar2015_textdet_test
]

# # Use ConcatDataset to combine the datasets in the list
train_dataset = dict(
       type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
       type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)


train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
