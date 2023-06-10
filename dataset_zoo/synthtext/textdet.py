data_root = 'data/synthtext'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='magnet:?xt=urn:btih:2dba9518166cbd141534cbf381aa3e99a08'
                '7e83c&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&t'
                'r=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2F'
                'tracker.opentrackr.org%3A1337%2Fannounce',
                save_name='SynthText.zip',
                md5='d1abe2ca57a0833328af9ae34ae78034',
                split=['train'],
                content=['image', 'annotation'],
                mapping=[['SynthText/SynthText/train/*', 'textdet_imgs/train/'],
                         ['textdet_imgs/train/train_gt.mat', 'annotations/train_gt.mat']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='train_gt.mat'),
    parser=dict(type='SynthTextAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    files=[
        dict(
            url='magnet:?xt=urn:btih:2dba9518166cbd141534cbf381aa3e99a08'
            '7e83c&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&t'
            'r=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2F'
            'tracker.opentrackr.org%3A1337%2Fannounce',
            save_name='SynthText.zip',
            md5='d1abe2ca57a0833328af9ae34ae78034',
            split=['test'],
            content=['image', 'annotation'],
            mapping=[['SynthText/SynthText/test/*', 'textdet_imgs/test/'],
                        ['textdet_imgs/test/test_gt.mat', 'annotations/test_gt.mat']]),
    ]),
    gatherer=dict(type='MonoGatherer', ann_name='test_gt.mat'),
    parser=dict(type='SynthTextAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['SynthText', 'annotations']

# config_generator = dict(
#     type='TextDetConfigGenerator', data_root=data_root, test_anns=None)
config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)

