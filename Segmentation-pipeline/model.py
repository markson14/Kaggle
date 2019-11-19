import torch.nn.functional as F
import torch


def metric(probability_mask, truth_mask, use_reject=True):

    probability_label = F.adaptive_max_pool2d(probability_mask, 1).view(1, -1)
    truth_label = F.adaptive_max_pool2d(truth_mask, 1).view(1, -1)

    threshold_label = 0.60
    threshold_mask = 0.30
    threshold_size = 1

    with torch.no_grad():
        batch_size, num_class = truth_label.shape

        probability = probability_label.view(batch_size, num_class)
        truth = truth_label.view(batch_size, num_class)

        # ----
        lp = (probability > threshold_label).float()
        lt = (truth > 0.5).float()
        num_tp = lt.sum(0)
        num_tn = batch_size-num_tp

        # ----
        tp = ((lp + lt) == 2).float()  # True positives
        tn = ((lp + lt) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        # ----------------------------------------------------------
        batch_size, num_class, H, W = truth_mask.shape

        probability = probability_mask.view(batch_size, num_class, -1)
        truth = truth_mask.view(batch_size, num_class, -1)

        # ------
        mp = (probability > threshold_mask).float()
        mt = (truth > 0.5).float()
        mt_sum = mt.sum(-1)
        mp_sum = mp.sum(-1)

        neg_index = (mt_sum == 0).float()
        pos_index = 1-neg_index

        if use_reject:  # get subset
            neg_index = neg_index*lp
            pos_index = pos_index*lp

        num_dn = neg_index.sum(0)
        num_dp = pos_index.sum(0)

        # ------
        dn = (mp_sum < threshold_size).float()
        dp = 2*(mp*mt).sum(-1)/((mp+mt).sum(-1)+1e-12)
        dn = (dn*neg_index).sum(0)
        dp = (dp*pos_index).sum(0)

        # ----
        all = torch.cat([
            tn, tp, num_tn, num_tp,
            dn, dp, num_dn, num_dp,
        ])
        all = all.data.cpu().numpy().reshape(-1, num_class)
        tn, tp, num_tn, num_tp, dn, dp, num_dn, num_dp = all

    return tn, tp, num_tn, num_tp, dn, dp, num_dn, num_dp
