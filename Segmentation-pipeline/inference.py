from train import load_model
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, AUCCallback, IouCallback
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import cv2
import seaborn as sns
from dataset import post_process, mask2rle, dice, sigmoid, CloudDataset, get_validation_augmentation, get_preprocessing
import torch
import gc
import pandas as pd
from torch.utils.data import DataLoader


def optimal_valid(k, net, config, loader_fold, ENCODER, ENCODER_WEIGHTS, ACTIVATION):
    runner = SupervisedRunner()
    model = load_model(net, ENCODER, ENCODER_WEIGHTS, ACTIVATION)

    logdir = "./logs/segmentation_{}_{}Fold".format(net, k)
    loaders = {"infer": loader_fold[k]['valid']}

    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/{config}.pth"),
            InferCallback()
        ],
    )
    ###################### dummy test ######################
    if 1:
        label_list = ["Fish", "Flower", "Gravel", "Sugar"]
        valid_masks = []
        probabilities = np.zeros(
            (len(loader_fold[k]['valid'].dataset) * 4, 350, 525))
        for i, (batch, output) in tqdm.tqdm_notebook(enumerate(zip(
                loaders['infer'].dataset, runner.callbacks[0].predictions["logits"]))):
            image, mask = batch
            for m in mask:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350),
                                   interpolation=cv2.INTER_LINEAR)
                valid_masks.append(m)

            for j, probability in enumerate(output):
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(
                        525, 350), interpolation=cv2.INTER_LINEAR)
                probabilities[i * 4 + j, :, :] = probability

        # Find optimal values
        # First of all, my thanks to @samusram for finding a mistake in my validation
        # https://www.kaggle.com/c/understanding_cloud_organization/discussion/107711#622412
        # And now I find optimal values separately for each class.

        class_params = {}
        for class_id in range(4):
            print(label_list[class_id])
            attempts = []
            for t in range(0, 100, 5):
                t /= 100
                for ms in [0, 100, 1200, 5000, 10000]:
                    masks = []
                    for i in range(class_id, len(probabilities), 4):
                        probability = probabilities[i]
                        predict, num_predict = post_process(
                            sigmoid(probability), t, ms)
                        masks.append(predict)

                    d = []
                    for i, j in zip(masks, valid_masks[class_id::4]):
                        if (i.sum() == 0) & (j.sum() == 0):
                            d.append(1)
                        else:
                            d.append(dice(i, j))

                    attempts.append((t, ms, np.mean(d)))

            attempts_df = pd.DataFrame(
                attempts, columns=['threshold', 'size', 'dice'])

            attempts_df = attempts_df.sort_values('dice', ascending=False)
            print(attempts_df.head())
            best_threshold = attempts_df['threshold'].values[0]
            best_size = attempts_df['size'].values[0]
            best_dice = attempts_df['dice'].values[0]

            class_params[class_id] = (best_threshold, best_size, best_dice)

        print("Best Threshold", class_params)
        print()
        print("Avg Valid Dice", (class_params[0][2] + class_params[1]
                                 [2] + class_params[2][2] + class_params[3][2])/4)
    else:
        class_params = {0: (0.6, 10000, 0.614792005689229),
                        1: (0.7, 10000, 0.7479094686835059),
                        2: (0.55, 10000, 0.6083618093569516),
                        3: (0.45, 10000, 0.5766765025111799)}

    ###################### dummy test ######################

    # print("Classification Report")

    del loaders
    torch.cuda.empty_cache()
    gc.collect()

    return class_params, runner


def infer(class_params, runner, infer_loaders, tta, pixels_tta):
    encoded_pixels = []
    image_id = 0
    emp = np.zeros((350, 525))
    for i, test_batch in tqdm.tqdm_notebook(enumerate(infer_loaders)):
        runner_out = runner.predict_batch(
            {"features": test_batch[0].cuda()})['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:
                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(
                        525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(
                    probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append(emp)
                else:
                    if tta == "flipv":
                        pixels_tta[image_id] += np.flip(predict, 0)
                    elif tta == "fliph":
                        pixels_tta[image_id] += np.flip(predict, 1)
                    elif tta == "normal":
                        pixels_tta[image_id] += predict
                    encoded_pixels.append(pixels_tta[image_id])
                # return the pure prediction
                # encoded_pixels.append(sigmoid(probability))
                image_id += 1

    return encoded_pixels, pixels_tta


def infer_pipeline(parameters):

    sub = parameters['sub']
    k = parameters["k"]
    net = parameters["net"]
    config = parameters["config"]
    loader_fold = parameters["loader_fold"]
    test_ids = parameters["test_ids"]
    rsize = parameters["rsize"]
    preprocessing_fn = parameters["preprocessing_fn"]
    avg_dice = parameters["avg_dice"]
    ENCODER = parameters["ENCODER"]
    ENCODER_WEIGHTS = parameters["ENCODER_WEIGHTS"]
    ACTIVATION = parameters["ACTIVATION"]

    print()
    print('#' * 10, 'Validating FOLD', k, '#' * 10)
    class_params, runner = optimal_valid(
        k, net, config, loader_fold, ENCODER, ENCODER_WEIGHTS, ACTIVATION)

    print()
    print('#' * 10, 'Infering FOLD', k, '#' * 10)

    tta_list = ["normal", "flipv", "fliph"]
    # tta_list = ["normal"]
    pixels_tta = [np.zeros((350, 525)) for _ in range(len(sub))]
    for tta in tta_list:
        print("TTA: ", tta)
        test_dataset = CloudDataset(df=sub,
                                    datatype='test',
                                    img_ids=test_ids,
                                    transforms=get_validation_augmentation(
                                        rsize, tta),
                                    preprocessing=get_preprocessing(preprocessing_fn))
        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=32)
        encoded_pixels, pixels_tta = infer(
            class_params, runner, test_loader, tta, pixels_tta)

    avg_dice += (class_params[0][2] + class_params[1]
                 [2] + class_params[2][2] + class_params[3][2]) / 4

    return encoded_pixels, avg_dice

# def get_label(probabilities):
#     label_list = []
#     for i in range(0, len(probabilities), 4):
#         fish = probabilities[i]
#         flower = probabilities[i+1]
#         grave = probabilities[i+2]
#         sugar = probabilities[i+3]

#         fish_label = (fish.sum(0).sum(0) > 0).astype(np.float32)
#         flower_label = (fish.sum(0).sum(0) > 0).astype(np.float32)
#         grave_label = (fish.sum(0).sum(0) > 0).astype(np.float32)
#         sugar_label = (fish.sum(0).sum(0) > 0).astype(np.float32)


if __name__ == "__main__":
    pass
