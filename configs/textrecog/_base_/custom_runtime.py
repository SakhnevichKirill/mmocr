
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),    
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        by_epoch=True, 
        max_keep_ckpts=2, 
        save_best=[
            'icdar2015/recog/word_acc', 
            # 'synthtext/recog/word_acc', 
            # 'synthtext/recog/word_acc_ignore_case'
        ], 
        rule=[
            'greater', 
            # 'greater', 
            # 'greater'
        ],
        save_optimizer=True,
        save_param_scheduler=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1000,
        enable=False,
        show=False,
        draw_gt=True,
        draw_pred=True),
)

# Logging
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

load_from = None
resume = False


# Evaluation
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric'),
    ],
    dataset_prefixes=None)

train_evaluator = val_evaluator

test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(tags=["recog"] ))]
visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends,
)

tta_model = dict(type='EncoderDecoderRecognizerTTAModel')