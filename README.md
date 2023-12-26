# Dogs vs. Cats Redux: Kernels Edition

ROBT407 Course Project

by Aliza Momysheva, Ariana Sadyr

## Desciption

To achieve sufficient accuracy in the classification of dogs versus cats kaggle competition, transfer learning, particularly pre-trained on the ImageNet dataset EfficientNetV2S was used in this project. The lowest BCE loss we could achieve on the validation set was 0.0199, which secured us on 160s place on the leaderboard with the loss of 0.06894 on the testing set. We have proved that MLP is not a good choice when we are working with complex images as we could only achieve around 65% accuracy on this task.

## Datasets

1. [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview)

### Prepare Dataset

Download dataset from links above, unzip and put them to dataset folder.

Use `dataset_prepare.ipynb` to divide the dataset to train and test

## Learning and Configuration

File to be changed is `config.yaml`. Best configurations we had:

1. `LEARNING_RATE`: 1e-3
2. `LEARNING_SCHEDULER`: ReduceLROnPlateau
3. `BATCH_SIZE`: 512
4. `PATIENCE`: 5
5. `NUM_EPOCHS` : 100
6. `IMAGE_SIZE` : 224
7. `MODEL`: EfficientNetV2S
8. `PRETRAINED`: True
9. `LOSS`: BCEWithLogitsLoss

To start learning:

`$ python train.py`

## Submission

Get the `run` ID under `runs` folder. Put it in the `test_submission.ipynb` file and run it. Submit `submission.csv` to Kaggle and see results.
