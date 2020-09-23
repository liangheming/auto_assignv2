import torch

from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from commons.model_utils import rand_seed


# rand_seed(1024)


def fcos_temp():
    from nets.fcos import FCOS
    from losses.auto_assign_loss import FCOSAutoAssignLoss
    dataset = COCODataSets(img_root="/home/thunisoft-root/liangheming/data/coco/coco2017/images/val2017",
                           annotation_path="/home/thunisoft-root/liangheming/data/coco/coco2017/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           img_size=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    net = FCOS(
        backbone="resnet50"
    )
    creterion = FCOSAutoAssignLoss()
    for img_input, targets, _ in dataloader:
        _, _, h, w = img_input.shape
        targets[:, 3:] = targets[:, 3:] * torch.tensor(data=[w, h, w, h])
        cls_outputs, reg_outputs, implicit_outputs, grids, gaussian = net(img_input)
        creterion(cls_outputs, reg_outputs, implicit_outputs, grids, gaussian, targets)
        # total_loss, detail_loss = creterion(cls_outputs, reg_outputs, center_outputs, grids,
        #                                     targets)
        # cls_loss, reg_loss, center_loss = detail_loss
        # print(total_loss)
        # print(cls_loss, reg_loss, center_loss)
        break


def download_weights():
    from torchvision.models.resnet import resnet50
    resnet50(pretrained=True)


if __name__ == '__main__':
    fcos_temp()
