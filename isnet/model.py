from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modules import RSU4F, RSU4, RSU5, RSU6, RSU7, _upsample_like


class ISNet(nn.Module):

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(ISNet, self).__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return (
            [F.sigmoid(d1),
             F.sigmoid(d2),
             F.sigmoid(d3),
             F.sigmoid(d4),
             F.sigmoid(d5),
             F.sigmoid(d6)],
            [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
        )


def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor]:
    # bce_loss = nn.BCELoss(size_average=True)
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        if preds[i].shape[2] != targets.shape[2] or preds[i].shape[3] != targets.shape[3]:
            tmp_target = F.interpolate(targets, size=preds[i].size()[
                                       2:], mode='bilinear', align_corners=True)
            loss = loss + \
                F.binary_cross_entropy_with_logits(preds[i], tmp_target)
        else:
            loss = loss + F.binary_cross_entropy_with_logits(preds[i], targets)
        if i == 0:
            loss0 = loss
    return loss0, loss


class ISNetModule(pl.LightningModule):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        self.net = ISNet(in_ch, out_ch)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        images, labels = train_batch
        outputs = self.net(images)
        loss2, loss = compute_loss(outputs, labels)
        self.log('train_loss2', loss2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int):
        images, labels = val_batch
        outputs = self.net(images)
        loss2, loss = compute_loss(outputs, labels)
        self.log('val_loss2', loss2)
        self.log('val_loss', loss)
