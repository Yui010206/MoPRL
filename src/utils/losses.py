import torch
import torch.nn as nn
import numpy as np
import math

class OKS_Loss(nn.Module):
    def __init__(self):
        super(OKS_Loss, self).__init__()
        
    def forward(self, predicted_pose, target_pose, weight=None):
        # predicted: B,N,2
        # mask: B, N
        # weitgt: B, N

        assert predicted_pose.shape == target_pose.shape

        norm_pose = torch.norm((predicted_pose - target_pose), p=2, dim=-1)
        if weight is not None:
            norm_pose = norm_pose.clone() * weight

        loss = norm_pose.mean()
        return loss 

class IOU_Loss(nn.Module):
    def __init__(self):
        super(IOU_Loss, self).__init__()

    def forward(self, predict_, target_, eps=1e-7):
        """`Implementation of Distance-IoU Loss: Faster and Better
        Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

        Code is modified from https://github.com/Zzh-tju/DIoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).
        Return:
            Tensor: Loss tensor.
        """
        # overlap
        assert predict_.shape == target_.shape

        pre_xy_min = predict_[:,::2,:]
        pre_xy_max = predict_[:,1::2,:]

        gt_xy_min = target_[:,::2,:]
        gt_xy_max = target_[:,1::2,:]

        pred = torch.cat([pre_xy_min,pre_xy_max],dim=-1).reshape(-1,4)
        target = torch.cat([gt_xy_min,gt_xy_max],dim=-1).reshape(-1,4)

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        loss = 1 - ious

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 1.), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.), loss)
        loss = loss.mean()

        return loss


class GIOU_Loss(nn.Module):
    def __init__(self):
        super(GIOU_Loss, self).__init__()

    def forward(self, predict_, target_, eps=1e-7):
        """`Implementation of Distance-IoU Loss: Faster and Better
        Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

        Code is modified from https://github.com/Zzh-tju/DIoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).
        Return:
            Tensor: Loss tensor.
        """
        # overlap
        assert predict_.shape == target_.shape

        pre_xy_min = predict_[:,::2,:]
        pre_xy_max = predict_[:,1::2,:]

        gt_xy_min = target_[:,::2,:]
        gt_xy_max = target_[:,1::2,:]

        pred = torch.cat([pre_xy_min,pre_xy_max],dim=-1).reshape(-1,4)
        target = torch.cat([gt_xy_min,gt_xy_max],dim=-1).reshape(-1,4)

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        loss = 1 - ious

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 1.), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.), loss)
        loss = loss.mean()

        return loss

class DIOU_Loss(nn.Module):
    def __init__(self):
        super(DIOU_Loss, self).__init__()

    def forward(self, predict_, target_, eps=1e-7):
        """`Implementation of Distance-IoU Loss: Faster and Better
        Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

        Code is modified from https://github.com/Zzh-tju/DIoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).
        Return:
            Tensor: Loss tensor.
        """
        # overlap
        assert predict_.shape == target_.shape

        pre_xy_min = predict_[:,::2,:]
        pre_xy_max = predict_[:,1::2,:]

        gt_xy_min = target_[:,::2,:]
        gt_xy_max = target_[:,1::2,:]

        pred = torch.cat([pre_xy_min,pre_xy_max],dim=-1).reshape(-1,4)
        target = torch.cat([gt_xy_min,gt_xy_max],dim=-1).reshape(-1,4)

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]
        
        # 最小包闭区域的对角线距离
        c2 = cw**2 + ch**2 + eps

        b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
        b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]
     
        # 中心点距离
        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4  #== ((b2_x1 + b2_x2)/2 - (b1_x1 + b1_x2)/2)**2
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4 
        rho2 = left + right           

        # DIoU
        dious = ious - rho2 / c2
        loss = 1 - dious

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 1.), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.), loss)
        loss = loss.mean()

        return loss

class CIOU_Loss(nn.Module):
    def __init__(self):
        super(CIOU_Loss, self).__init__()

    def forward(self, predict, target, eps=1e-7):
        predict_ = predict.clone()
        target_ = target.clone()

        assert predict_.shape == target_.shape

        pre_xy_min = predict_[:,::2,:]
        pre_xy_max = predict_[:,1::2,:]

        gt_xy_min = target_[:,::2,:]
        gt_xy_max = target_[:,1::2,:]

        pred = torch.cat([pre_xy_min,pre_xy_max],dim=-1).reshape(-1,4)
        target = torch.cat([gt_xy_min,gt_xy_max],dim=-1).reshape(-1,4)

        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw**2 + ch**2 + eps

        b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
        b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
        rho2 = left + right
        # 对应公式
        factor = 4 / math.pi**2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        # CIoU
        cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
        loss = 1 - cious

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 1.), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.), loss)
        loss = loss.mean()

        return loss

if __name__ == '__main__':
    loss = DIOU_Loss()

    predicted = torch.randint(1,10,(2,16,2))
    target = torch.randint(1,10,(2,16,2))

    l = loss(predicted,target)

    print(l)