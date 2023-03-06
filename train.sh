DATA_PATH=${1:-/dataset/coco2017}
GPU_NUMS=${2:-1}
# MODEL = ssd300_vgg16, fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
MODEL=${3:-ssd300_vgg16}
BATCH_SIZE=${4:-16}
PRINT_FRREP=100
LR=0.0010

python3 -m oneflow.distributed.launch \
  --nproc_per_node $GPU_NUMS \
  --nnodes 1 \
  --master_port 12345 \
  --master_addr 127.0.0.1 \
  train.py \
    --data-path $DATA_PATH \
    --epochs 1 \
    --evaluation 2 \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --print-freq $PRINT_FRREP \
    --world-size $GPU_NUMS \
    --lr $LR
