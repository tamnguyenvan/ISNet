import torch


def mae_torch(pred, gt):
    h, w = gt.shape[0:2]
    sum_error = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
    mae_error = torch.divide(sum_error, float(h)*float(w)*255.0+1e-4)
    return mae_error


def f1score_torch(pd, gt):
    gtNum = torch.sum((gt > 128).float()*1)  # number of ground truth pixels

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = torch.histc(pp, bins=255, min=0, max=255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255)

    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum) / \
        (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = (pp_hist_flip_cum) / (gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall / (0.3*precision+recall + 1e-4)

    return (
        torch.reshape(precision, (1, precision.shape[0])),
        torch.reshape(recall, (1, recall.shape[0])),
        torch.reshape(f1, (1, f1.shape[0]))
    )


def f1_mae_torch(pred, gt):
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    pre, rec, f1 = f1score_torch(pred, gt)
    mae = mae_torch(pred, gt)

    return pre, rec, f1, mae
