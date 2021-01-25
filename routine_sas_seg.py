import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
from torch.utils.data import DataLoader
import cv2

from dataset_sas_seg import visualize, get_training_augmentation, get_validation_augmentation, get_preprocessing, \
                                    SegDataset

class SasSegmentation:
    def __init__(self, ENCODER = 'se_resnext50_32x4d',
                 ENCODER_WEIGHTS = 'imagenet',
                 CLASSES = ['suitable'],
                 ACTIVATION = 'sigmoid',
                 DEVICE = 'cuda'):

        self.CLASSES = CLASSES
        self.DEVICE = DEVICE
        self.model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        self.optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=0.0001),
        ])

    def init_data(self, DATA_DIR=f'seg_images/'):

        self.DATA_DIR = DATA_DIR
        x_train_dir = os.path.join(DATA_DIR, 'train')
        y_train_dir = os.path.join(DATA_DIR, 'train_annot')

        x_valid_dir = os.path.join(DATA_DIR, 'val')
        y_valid_dir = os.path.join(DATA_DIR, 'val_annot')


        train_dataset = SegDataset(
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.CLASSES,
        )

        valid_dataset = SegDataset(
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.CLASSES,
        )

        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    def run(self, num_epochs=10):
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.DEVICE,
            verbose=True,
        )

        max_score = 0
        for i in range(0, num_epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.train_loader)
            valid_logs = valid_epoch.run(self.valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, './best_model.pth')
                print('Model saved!')

            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

    def test(self, vis=False):

        x_test_dir = os.path.join(self.DATA_DIR, 'test')
        y_test_dir = os.path.join(self.DATA_DIR, 'test_annot')
        # create test dataset
        test_dataset = SegDataset(
            x_test_dir,
            y_test_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.CLASSES,
        )

        test_dataset_vis = SegDataset(
            x_test_dir, y_test_dir,
            classes=self.CLASSES,
        )

        test_dataloader = DataLoader(test_dataset)

        best_model = torch.load('./best_model.pth')
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.DEVICE,
        )

        logs = test_epoch.run(test_dataloader)

        if vis:
            for n in range(2):
                # n = np.random.choice(len(test_dataset))

                image_vis = test_dataset_vis[n][0].astype('uint8')
                image, gt_mask = test_dataset[n]

                gt_mask = gt_mask.squeeze()

                x_tensor = torch.from_numpy(image).to(self.DEVICE).unsqueeze(0)
                pr_mask = best_model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())

                visualize(
                    image=image_vis,
                    ground_truth_mask=gt_mask,
                    predicted_mask=pr_mask
                )


if __name__ == "__main__":
    SegNet = SasSegmentation()
    SegNet.init_data()
    SegNet.run(num_epochs=3)
    SegNet.test(vis=True)
