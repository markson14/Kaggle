import numpy as np
import pandas as pd
import albumentations as albu
import os
import cv2
from torch.utils.data import Dataset


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def sigmoid(x): return 1 / (1 + np.exp(-x))


def get_training_augmentation(rsize):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5,
                              rotate_limit=20,
                              shift_limit=0.1,
                              p=0.5,
                              border_mode=0)]
    if rsize == 320:
        train_transform.append(albu.Resize(320, 480))
    elif rsize == 384:
        train_transform.append(albu.Resize(384, 576))
    elif rsize == 640:
        train_transform.append(albu.Resize(640, 960))
    elif rsize == 256:
        train_transform.append(albu.Resize(256, 384))
    elif rsize == 700:
        train_transform.append(albu.Resize(700, 1050))
    elif rsize == 64:
        train_transform.append(albu.Resize(64, 96))

    return albu.Compose(train_transform)


def get_validation_augmentation(rsize, tta):
    """Add paddings to make image shape divisible by 32"""
    test_transform = []
    if rsize == 320:
        test_transform.append(albu.Resize(320, 480))
    elif rsize == 384:
        test_transform.append(albu.Resize(384, 576))
    elif rsize == 640:
        test_transform.append(albu.Resize(640, 960))
    elif rsize == 256:
        test_transform.append(albu.Resize(256, 384))
    elif rsize == 700:
        test_transform.append(albu.Resize(700, 1050))
    elif rsize == 64:
        test_transform.append(albu.Resize(64, 96))

    if tta == 'flipv':
        test_transform.append(albu.VerticalFlip(p=1))
    elif tta == 'fliph':
        test_transform.append(albu.HorizontalFlip(p=1))
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(),
                                          albu.VerticalFlip(),
                                          albu.ShiftScaleRotate()]),
                 preprocessing=None, path="."):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)[::2, ::2, :]
#         mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)[::2, ::2, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


if __name__ == "__main__":
    pass
