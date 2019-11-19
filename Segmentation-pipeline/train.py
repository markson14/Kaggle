from torch import nn
import torch
from radam import RAdam
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, AUCCallback, IouCallback
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import gc


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return 100*torch.mean(F_loss)
        else:
            return 100*F_loss


class FocalDiceLoss(smp.utils.losses.DiceLoss):
    __name__ = 'focal_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.focal = FocalLoss()

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        focal = self.focal(y_pr, y_gt)
        return dice + focal


def load_model(net, ENCODER, ENCODER_WEIGHTS, ACTIVATION):
    if net == "Unet":
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=4,
            activation=ACTIVATION,
        )
    elif net == "FPN":
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=4,
            activation=ACTIVATION,
        )
    elif net == "PSPNet":
        model = smp.PSPNet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=4,
            activation=ACTIVATION,
        )
    return model


def train_model(train_parameters):

    k = train_parameters["k"]
    loaders = train_parameters["loaders"]
    num_epochs = train_parameters["num_epochs"]
    net = train_parameters["net"]
    ENCODER = train_parameters["ENCODER"]
    ENCODER_WEIGHTS = train_parameters["ENCODER_WEIGHTS"]
    ACTIVATION = train_parameters["ACTIVATION"]

    model = load_model(
        net, ENCODER, ENCODER_WEIGHTS, ACTIVATION)

    """ multi-gpu """
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to("cuda")

#     if k==0:
#         summary(model.module.encoder,(3,384,576))

    logdir = "./logs/segmentation_{}_{}Fold".format(net, k)

    # model, criterion, optimizer
    optimizer = RAdam([
        {'params': model.module.decoder.parameters(), 'lr': 1e-2},
        {'params': model.module.encoder.parameters(), 'lr': 1e-3},
        #         {'params': model.decoder.parameters(), 'lr': 1e-2},
        #         {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])

    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
#     criterion = FocalLoss()
#     criterion = FocalDiceLoss()
    # criterion = smp.utils.losses.DiceLoss(eps=1.)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[EarlyStoppingCallback(patience=10, min_delta=0.001),
                   DiceCallback()],
        #                    AUCCallback(),
        #                    IouCallback()],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    del loaders, optimizer, scheduler, model, runner
    torch.cuda.empty_cache()
    gc.collect()
    print("Collect GPU cache")


if __name__ == "__main__":
    pass
