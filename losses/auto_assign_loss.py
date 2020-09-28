import torch
from utils.fcos import BoxCoder
from losses.commons import IOULoss
from commons.boxs_utils import box_iou


class FCOSAutoAssignLoss(object):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 lambda_p=5.0,
                 temperature=1. / 3,
                 strides=None,
                 iou_type='giou',
                 positive_weights=0.1,
                 negative_weights=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.temperature = temperature
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights
        if strides is None:
            strides = [8, 16, 32, 64, 128]
        self.strides = strides
        self.box_coder = BoxCoder()
        self.iou_loss_func = IOULoss(iou_type=iou_type, coord_type='ltrb')

    def __call__(self, cls_predicts, box_predicts, implicits, grids, gaussian, targets):
        """
        :param cls_predicts: list(cls_predict) cls_predict [bs, cls, h, w]
        :param box_predicts: list(box_predict) box_predict [bs, 4, h, w]
        :param implicits: list(implicit) implicit[bs, 1, h, w]
        :param grids: [h, w, 2]
        :param gaussian: [cls, 4]
        :param targets: [gt, 7] (bs, weights, label_id, x1, y1, x2, y2)
        :return:
        """
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        cls_num = cls_predicts[0].shape[1]
        # expand_grid [grid_num,3](xc,yc,stride)
        expand_grid = torch.cat([
            torch.cat([
                grid_item,
                torch.tensor(data=stride_item, device=device, dtype=torch.float).expand_as(grid_item[..., [0]])
            ], dim=-1).view(-1, 3) for stride_item, grid_item in zip(self.strides, grids)], dim=0)
        for i in range(len(cls_predicts)):
            if cls_predicts[i].dtype == torch.float16:
                cls_predicts[i] = cls_predicts[i].float()
        for i in range(len(implicits)):
            if implicits[i].dtype == torch.float16:
                implicits[i] = implicits[i].float()
        negative_loss_list = list()
        positive_loss_list = list()
        for bi in range(bs):
            # batch_cls_predicts [grid_num,cls_num]
            batch_cls_predicts = torch.cat(
                [cls_item[bi].permute(1, 2, 0).contiguous().view(-1, cls_num) for cls_item in cls_predicts],
                dim=0).sigmoid()
            # batch_implicit [grid_num,1]
            batch_implicit = torch.cat(
                [implicit_item[bi].permute(1, 2, 0).contiguous().view(-1, 1) for implicit_item in implicits],
                dim=0).sigmoid()

            batch_join_predicts = (batch_cls_predicts * batch_implicit).clamp(1e-6, 1 - 1e-6)
            # batch_box_predicts [grid_num, 4]
            batch_box_predicts = torch.cat(
                [box_item[bi].permute(1, 2, 0).contiguous().view(-1, 4) for box_item in box_predicts], dim=0)
            batch_targets = targets[targets[:, 0] == bi, 1:]
            if len(batch_targets) == 0:
                negative_loss = -(1 - self.alpha) * batch_join_predicts ** self.gamma * (
                        1 - batch_join_predicts).log()
                negative_loss = negative_loss.sum()
                negative_loss_list.append(negative_loss)
                continue
            # [gt_num,6] (weights,label_idx,x1,y1,x2,y2)
            gt_xy = (batch_targets[:, [2, 3]] + batch_targets[:, [4, 5]]) / 2
            # [grid_num,gt_num,2]
            xy_offset = (expand_grid[:, None, :2] - gt_xy[None, :, :]) / expand_grid[:, None, [2]]
            # [grid_num,gt_num,4]
            batch_reg_targets = self.box_coder.encode(expand_grid[..., :2], batch_targets[..., 2:])
            grid_idx, gt_idx = (batch_reg_targets.min(dim=-1)[0] > 0).nonzero(as_tuple=False).t()

            cls_prob = batch_join_predicts[grid_idx, batch_targets[gt_idx, 1].long()]
            iou_loss = self.iou_loss_func(batch_box_predicts[grid_idx, :], batch_reg_targets[grid_idx, gt_idx, :])
            loc_prob = (-self.lambda_p * iou_loss).exp()
            joint_prob = cls_prob * loc_prob
            confidence = (joint_prob / self.temperature).exp()
            gaussian_delta_mu = -(
                    (xy_offset[grid_idx, gt_idx, :] - gaussian[batch_targets[gt_idx, 1].long(), :2]) ** 2
            ).sum(-1)
            gaussian_delta_theta = 2 * ((gaussian[batch_targets[gt_idx, 1].long(), 2:]) ** 2).sum(-1)
            gaussian_weights = (gaussian_delta_mu / gaussian_delta_theta).exp()
            positive_weights = confidence * gaussian_weights
            positive_loss = torch.tensor(data=0., device=device)
            for unique_gt_idx in gt_idx.unique():
                grid_idx_mask = gt_idx == unique_gt_idx
                instance_weights = positive_weights[grid_idx_mask] / positive_weights[grid_idx_mask].sum()
                instance_loss = -(instance_weights * joint_prob[grid_idx_mask]).sum().log()
                positive_loss += instance_loss
            positive_loss_list.append(positive_loss)

            decode_box = self.box_coder.decoder(expand_grid[..., :2], batch_box_predicts).detach()
            predict_targets_iou = box_iou(decode_box, batch_targets[..., 2:])
            max_iou, max_iou_gt_idx = predict_targets_iou.max(dim=-1)
            func_iou = 1 / (1 - max_iou)
            func_iou = 1 - (func_iou - 1) / (func_iou.max() - 1 + 1e-10)
            negative_weights = torch.ones(size=(expand_grid.shape[0], cls_num), device=device).float()
            negative_weights[grid_idx, batch_targets[gt_idx, 1].long()] = func_iou[grid_idx]
            weighted_negative_prob = negative_weights * batch_join_predicts
            negative_loss = -(1 - self.alpha) * weighted_negative_prob ** self.gamma * (
                    1 - weighted_negative_prob).log()
            negative_loss = negative_loss.sum()
            negative_loss_list.append(negative_loss)
        total_negative_loss = torch.stack(negative_loss_list).sum() / max(1, len(targets))
        if len(targets) == 0:
            return total_negative_loss, \
                   torch.stack([total_negative_loss, torch.tensor(0., device=device)]).detach(), \
                   len(targets)
        total_positive_loss = torch.stack(positive_loss_list).sum() / max(1, len(targets))
        total_negative_loss = total_negative_loss * self.negative_weights
        total_positive_loss = total_positive_loss * self.positive_weights
        total_loss = total_negative_loss + total_positive_loss
        return total_loss, torch.stack([total_negative_loss, total_positive_loss]).detach(), len(targets)


class FCOSAutoAssignLossPerInstance(object):
    def __init__(self, alpha=0.25,
                 gamma=2.0,
                 lambda_p=5.0,
                 temperature=1. / 3,
                 strides=None,
                 iou_type='giou',
                 positive_weights=1.0,
                 negative_weights=1.0
                 ):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.temperature = temperature
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights
        if strides is None:
            strides = [8, 16, 32, 64, 128]
        self.strides = strides
        self.box_coder = BoxCoder()
        self.iou_loss_func = IOULoss(iou_type=iou_type, coord_type='ltrb')

    def __call__(self, cls_predicts, box_predicts, implicits, grids, gaussian, targets):
        """
        :param cls_predicts: list(cls_predict) cls_predict [bs, cls, h, w]
        :param box_predicts: list(box_predict) box_predict [bs, 4, h, w]
        :param implicits: list(implicit) implicit[bs, 1, h, w]
        :param grids: [h, w, 2]
        :param gaussian: [cls, 4]
        :param targets: [gt, 7] (bs, weights, label_id, x1, y1, x2, y2)
        :return:
        """
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        cls_num = cls_predicts[0].shape[1]
        expand_grid = torch.cat([
            torch.cat([
                grid_item,
                torch.tensor(data=stride_item, device=device, dtype=torch.float).expand_as(grid_item[..., [0]])
            ], dim=-1).view(-1, 3) for stride_item, grid_item in zip(self.strides, grids)], dim=0)
        for i in range(len(cls_predicts)):
            if cls_predicts[i].dtype == torch.float16:
                cls_predicts[i] = cls_predicts[i].float()
        for i in range(len(implicits)):
            if implicits[i].dtype == torch.float16:
                implicits[i] = implicits[i].float()
        # expand_grid [grid_num,3](xc,yc,stride)
        negative_loss_list = list()
        positive_loss_list = list()
        for bi in range(bs):
            batch_cls_predicts = torch.cat(
                [cls_item[bi].permute(1, 2, 0).contiguous().view(-1, cls_num) for cls_item in cls_predicts],
                dim=0).sigmoid()
            batch_implicit = torch.cat(
                [implicit_item[bi].permute(1, 2, 0).contiguous().view(-1, 1) for implicit_item in implicits],
                dim=0).sigmoid()
            batch_join_predicts = (batch_cls_predicts * batch_implicit).clamp(1e-6, 1 - 1e-6)
            batch_box_predicts = torch.cat(
                [box_item[bi].permute(1, 2, 0).contiguous().view(-1, 4) for box_item in box_predicts], dim=0)
            batch_targets = targets[targets[:, 0] == bi, 1:]
            if len(batch_targets) == 0:
                negative_loss = -(1 - self.alpha) * batch_join_predicts ** self.gamma * (
                        1 - batch_join_predicts).log()
                negative_loss = negative_loss.sum()
                negative_loss_list.append(negative_loss)
                continue
            # [gt_num,6] (weights,label_idx,x1,y1,x2,y2)
            gt_xy = (batch_targets[:, [2, 3]] + batch_targets[:, [4, 5]]) / 2
            # [grid_num,gt_num,2]
            xy_offset = (expand_grid[:, None, :2] - gt_xy[None, :, :]) / expand_grid[:, None, [2]]
            # [grid_num,gt_num,4]
            batch_reg_targets = self.box_coder.encode(expand_grid[..., :2], batch_targets[..., 2:])
            grid_idx, gt_idx = (batch_reg_targets.min(dim=-1)[0] > 0).nonzero(as_tuple=False).t()

            cls_prob = batch_join_predicts[grid_idx, batch_targets[gt_idx, 1].long()]
            iou_loss = self.iou_loss_func(batch_box_predicts[grid_idx, :], batch_reg_targets[grid_idx, gt_idx, :])
            loc_prob = (-self.lambda_p * iou_loss).exp()
            joint_prob = cls_prob * loc_prob
            confidence = (joint_prob / self.temperature).exp()
            gaussian_delta_mu = -(
                    (xy_offset[grid_idx, gt_idx, :] - gaussian[batch_targets[gt_idx, 1].long(), :2]) ** 2
            ).sum(-1)
            gaussian_delta_theta = 2 * ((gaussian[batch_targets[gt_idx, 1].long(), 2:]) ** 2).sum(-1)
            gaussian_weights = (gaussian_delta_mu / gaussian_delta_theta).exp()
            positive_weights = confidence * gaussian_weights

            decode_box = self.box_coder.decoder(expand_grid[..., :2], batch_box_predicts).detach()
            predict_targets_iou = box_iou(decode_box, batch_targets[..., 2:])
            max_iou, max_iou_gt_idx = predict_targets_iou.max(dim=-1)
            func_iou = 1 / (1 - max_iou)
            negative_weights = torch.ones(size=(expand_grid.shape[0], cls_num), device=device).float()
            positive_loss = torch.tensor(data=0., device=device)
            for unique_gt_idx in gt_idx.unique():
                grid_idx_mask = gt_idx == unique_gt_idx
                instance_weights = positive_weights[grid_idx_mask] / positive_weights[grid_idx_mask].sum()
                instance_loss = -(instance_weights * joint_prob[grid_idx_mask]).sum().log()
                positive_loss += instance_loss
                inner_grid_idx = grid_idx[grid_idx_mask]
                normalized_negative_weights = 1 - (func_iou[inner_grid_idx] - 1) / (
                        func_iou[inner_grid_idx].max() - 1 + 1e-10)
                ori_weight = negative_weights[inner_grid_idx, batch_targets[unique_gt_idx][0].long()]
                cur_weight = normalized_negative_weights
                new_weight = torch.where(cur_weight < ori_weight, cur_weight, ori_weight)
                negative_weights[inner_grid_idx, batch_targets[unique_gt_idx][0].long()] = new_weight
            positive_loss_list.append(positive_loss)
            weighted_negative_prob = negative_weights * batch_join_predicts
            negative_loss = -(1 - self.alpha) * weighted_negative_prob ** self.gamma * (
                    1 - weighted_negative_prob).log()
            negative_loss = negative_loss.sum()
            negative_loss_list.append(negative_loss)
        total_negative_loss = torch.stack(negative_loss_list).sum() / max(1, len(targets))
        if len(targets) == 0:
            return total_negative_loss, \
                   torch.stack([total_negative_loss, torch.tensor(0., device=device)]).detach(), \
                   len(targets)
        total_positive_loss = torch.stack(positive_loss_list).sum() / max(1, len(targets))
        total_positive_loss = total_positive_loss * self.positive_weights
        total_negative_loss = total_negative_loss * self.negative_weights
        total_loss = total_negative_loss + total_positive_loss
        return total_loss, torch.stack([total_negative_loss, total_positive_loss]).detach(), len(targets)


class FCOSAutoAssignLossFreeAnchor(object):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_p=5.0, temperature=1. / 3, strides=None, iou_type='giou'):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.temperature = temperature
        if strides is None:
            strides = [8, 16, 32, 64, 128]
        self.strides = strides
        self.box_coder = BoxCoder()
        self.iou_loss_func = IOULoss(iou_type=iou_type, coord_type='ltrb')

    def __call__(self, cls_predicts, box_predicts, implicits, grids, gaussian, targets):
        """
        :param cls_predicts: list(cls_predict) cls_predict [bs, cls, h, w]
        :param box_predicts: list(box_predict) box_predict [bs, 4, h, w]
        :param implicits: list(implicit) implicit[bs, 1, h, w]
        :param grids: [h, w, 2]
        :param gaussian: [cls, 4]
        :param targets: [gt, 7] (bs, weights, label_id, x1, y1, x2, y2)
        :return:
        """
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        cls_num = cls_predicts[0].shape[1]
        expand_grid = torch.cat([
            torch.cat([
                grid_item,
                torch.tensor(data=stride_item, device=device, dtype=torch.float).expand_as(grid_item[..., [0]])
            ], dim=-1).view(-1, 3) for stride_item, grid_item in zip(self.strides, grids)], dim=0)
        for i in range(len(cls_predicts)):
            if cls_predicts[i].dtype == torch.float16:
                cls_predicts[i] = cls_predicts[i].float()
        for i in range(len(implicits)):
            if implicits[i].dtype == torch.float16:
                implicits[i] = implicits[i].float()
        # expand_grid [grid_num,3](xc,yc,stride)
        negative_loss_list = list()
        positive_loss_list = list()
        for bi in range(bs):
            batch_cls_predicts = torch.cat(
                [cls_item[bi].permute(1, 2, 0).contiguous().view(-1, cls_num) for cls_item in cls_predicts],
                dim=0).sigmoid()
            batch_implicit = torch.cat(
                [implicit_item[bi].permute(1, 2, 0).contiguous().view(-1, 1) for implicit_item in implicits],
                dim=0).sigmoid()
            batch_join_predicts = (batch_cls_predicts * batch_implicit).clamp(1e-6, 1 - 1e-6)
            batch_box_predicts = torch.cat(
                [box_item[bi].permute(1, 2, 0).contiguous().view(-1, 4) for box_item in box_predicts], dim=0)
            batch_targets = targets[targets[:, 0] == bi, 1:]
            if len(batch_targets) == 0:
                negative_loss = -(1 - self.alpha) * batch_join_predicts ** self.gamma * (
                        1 - batch_join_predicts).log()
                negative_loss = negative_loss.sum()
                negative_loss_list.append(negative_loss)
                continue
            # [gt_num,6] (weights,label_idx,x1,y1,x2,y2)
            gt_xy = (batch_targets[:, [2, 3]] + batch_targets[:, [4, 5]]) / 2
            # [grid_num,gt_num,2]
            xy_offset = (expand_grid[:, None, :2] - gt_xy[None, :, :]) / expand_grid[:, None, [2]]
            # [grid_num,gt_num,4]
            batch_reg_targets = self.box_coder.encode(expand_grid[..., :2], batch_targets[..., 2:])
            grid_idx, gt_idx = (batch_reg_targets.min(dim=-1)[0] > 0).nonzero(as_tuple=False).t()

            cls_prob = batch_join_predicts[grid_idx, batch_targets[gt_idx, 1].long()]
            iou_loss = self.iou_loss_func(batch_box_predicts[grid_idx, :], batch_reg_targets[grid_idx, gt_idx, :])
            loc_prob = (-self.lambda_p * iou_loss).exp()
            joint_prob = cls_prob * loc_prob
            confidence = (joint_prob / self.temperature).exp()
            gaussian_delta_mu = -(
                    (xy_offset[grid_idx, gt_idx, :] - gaussian[batch_targets[gt_idx, 1].long(), :2]) ** 2
            ).sum(-1)
            gaussian_delta_theta = 2 * ((gaussian[batch_targets[gt_idx, 1].long(), 2:]) ** 2).sum(-1)
            gaussian_weights = (gaussian_delta_mu / gaussian_delta_theta).exp()
            positive_weights = confidence * gaussian_weights

            positive_loss = torch.tensor(data=0., device=device)
            for unique_gt_idx in gt_idx.unique():
                grid_idx_mask = gt_idx == unique_gt_idx
                instance_weights = positive_weights[grid_idx_mask] / positive_weights[grid_idx_mask].sum()
                instance_loss = -(instance_weights * joint_prob[grid_idx_mask]).sum().log()
                positive_loss += instance_loss
            positive_loss_list.append(positive_loss)

            decode_box = self.box_coder.decoder(expand_grid[..., :2], batch_box_predicts).detach()
            target_box_iou = box_iou(batch_targets[..., 2:], decode_box)
            background_grid = torch.ones(size=(len(expand_grid),), dtype=torch.bool, device=device)
            background_grid[grid_idx.unique()] = False
            target_box_iou[:, background_grid] = 0.
            t1 = 0.1
            t2 = target_box_iou.max(dim=1, keepdim=True)[0].clamp(min=t1 + 1e-6)
            target_box_prob = ((target_box_iou - t1) / (t2 - t1)).clamp(min=0., max=1.)
            indices = torch.stack([torch.arange(len(batch_targets), device=device), batch_targets[:, 1]], dim=0).long()
            object_cls_box_prob = torch.sparse_coo_tensor(indices, target_box_prob, device=device)
            cls_idx, anchor_idx = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense().nonzero(as_tuple=False).t()
            if len(cls_idx) == 0:
                negative_loss = -(1 - self.alpha) * batch_join_predicts ** self.gamma * (
                        1 - batch_join_predicts).log()
                negative_loss = negative_loss.sum()
                negative_loss_list.append(negative_loss)
                continue
            anchor_positive_max_prob = torch.where(
                batch_targets[:, [1]].long() == cls_idx,
                target_box_prob[:, anchor_idx],
                torch.tensor(data=0., device=device)
            ).max(dim=0)[0]

            anchor_cls_assign_prob = torch.zeros(size=(len(expand_grid), cls_num), device=device)
            anchor_cls_assign_prob[anchor_idx, cls_idx] = anchor_positive_max_prob
            negative_prob = batch_join_predicts * (1 - anchor_cls_assign_prob)
            negative_loss = -(1 - self.alpha) * (negative_prob ** self.gamma) * (1 - negative_prob).log()
            negative_loss = negative_loss.sum()
            negative_loss_list.append(negative_loss)
        total_negative_loss = torch.stack(negative_loss_list).sum() / max(1, len(targets))
        if len(targets) == 0:
            return total_negative_loss, \
                   torch.stack([total_negative_loss, torch.tensor(0., device=device)]).detach(), \
                   len(targets)
        total_positive_loss = torch.stack(positive_loss_list).sum() / max(1, len(targets))
        total_loss = total_negative_loss + total_positive_loss
        return total_loss, torch.stack([total_negative_loss, total_positive_loss]).detach(), len(targets)
