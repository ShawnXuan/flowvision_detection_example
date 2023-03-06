OneFlow Flowvision Detection Examples.

Training with coco dataset
```bash
./train.sh [/path/to/coco/dataset] [#GPUS] [model_name] [batch_size]
```

Evaluation with coco dataset
```bash
./eval.sh [/path/to/coco/dataset] [#GPUS] [model_name] [batch_size]
```

model_names = [ssd300_vgg16, fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn]

## [coco dataset](https://cocodataset.org/#download)
Download coco2017 from links below
- 2017 Train images [118K/18GB]: http://images.cocodataset.org/zips/train2017.zip
- 2017 Val images [5K/1GB]: http://images.cocodataset.org/zips/val2017.zip
- 2017 Train/Val annotations [241MB]: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzipping
```bash
unzip *.zip
```

## pretraind models
Pretrained model will be automatically downloaded to $HOME/.oneflow/flowvision_cache directory.
