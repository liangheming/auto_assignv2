# auto_assign
This is an unofficial pytorch implementation of **Auto Assign** object detection as described in [AutoAssign: Differentiable Label Assignment for Dense Object Detection](https://arxiv.org/abs/2007.03496) by Benjin Zhu, Jianfeng Wang, Zhengkai Jiang, Fuhang Zong, Songtao Liu, Zeming Li, Jian Sun.
But we have some problem about the negative loss function. we implement total loss function according to your own understanding and there is an unbalance between the positive loss and the negative loss.
As a result we didn't achieve a satisfactory performance 36.8 mAP which is less than our fcos implementation 37.5 mAP.
## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.6
torchvision >=0.7.0
```
## result
we trained this repo on 2 GPUs with batch size 16(8 image per node).the total epoch is 24(about 180k iter),SGD with cosine lr decay is used for optimizing.
finally, this repo achieves 36.8 mAp at 768px(max side) resolution with resnet50 backbone.
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.572
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.386
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.492
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.493
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.536
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.353
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.574
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.672
```
## difference from original implement
the main difference is about the input resolution.the original implement use *min_thresh* and *max_thresh* to keep the short
side of the input image larger than *min_thresh* while keep the long side smaller than *max_thresh*.for simplicity we fix the long
side a certain size, then we resize the input image while **keep the width/height ratio**, next we pad the short side.the final
width and height of the input are same.
## training
for now we only support coco detection data.
### COCO
* modify main.py (modify config file path)
```python
from processors.ddp_mix_processor import DDPMixProcessor
if __name__ == '__main__':
    processor = DDPMixProcessor(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: retinanet
data:
  train_annotation_path: ../annotations/instances_train2017.json 
  val_annotation_path: ../annotations/instances_val2017.json
  train_img_root: ../data/train2017
  val_img_root: ../data/val2017
  img_size: 768
  use_crowd: False
  batch_size: 8
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  strides: [8, 16, 32, 64, 128]
  backbone: resnet50
  freeze_bn: False

hyper_params:
  alpha: 0.25
  gamma: 2.0
  multi_scale: [768]
  layer_limits: [64, 128, 256, 512]
  radius: 5
  iou_type: giou

optim:
  optimizer: SGD
  lr: 0.01
  momentum: 0.9
  milestones: [18,24]
  cosine_weights: 1.0
  warm_up_epoch: 1.
  weight_decay: 0.0001
  epochs: 25
  sync_bn: True
val:
  interval: 1
  weight_path: weights
  conf_thresh: 0.05
  iou_thresh: 0.5
  max_det: 300

gpus: 0,1
```

* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=2 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] Center Sample
- [x] EMA(Exponential Moving Average)
- [x] Mixed Precision Training (torch native amp)
- [x] Sync Batch Normalize
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support