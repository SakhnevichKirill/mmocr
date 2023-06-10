_base_ = 'default_runtime.py'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        by_epoch=True, 
        max_keep_ckpts=2, 
        save_best=['synthtext/det/hmean', 'icdar2015/det/hmean'], 
        rule=['greater', 'greater'],
        save_optimizer=True,
        save_param_scheduler=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1000,
        enable=True,
        show=False,
        draw_gt=True,
        draw_pred=True),
)

# # Evaluation
# val_evaluator = dict(type='HmeanIOUMetric')
# test_evaluator = val_evaluator

# Evaluation
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(type='HmeanIOUMetric', prefix='det'),
    ],
    dataset_prefixes=None)
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(tags=["det"] ))]

visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends)
