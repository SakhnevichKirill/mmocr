synthtext_textrecog_data_root = 'data/synthtext'

synthtext_textrecog_train = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

synthtext_textrecog_test = dict(
    type='OCRDataset',
    data_root=synthtext_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
