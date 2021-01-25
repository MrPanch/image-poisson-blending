from routine_sas_seg import SasSegmentation


if __name__ == '__main__':
    SegNet = SasSegmentation(ENCODER='se_resnext50_32x4d',
                 ENCODER_WEIGHTS='imagenet',
                 CLASSES=['suitable'],
                 ACTIVATION='sigmoid',
                 DEVICE='cuda')

    SegNet.init_data(DATA_DIR=f'seg_images/')

    SegNet.run(num_epochs=100)
    SegNet.test(vis=True)