DATA_PATH=${1:-/dataset/coco2017}
GPU_NUMS=${2:-1}
# MODEL = ssd300_vgg16, fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
MODEL=${3:-ssd300_vgg16}
BATCH_SIZE=${4:-16}
PRINT_FRREP=100
LR=0.0010

python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPU_NUMS \
    train.py \
    --data-path $DATA_PATH \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --world-size $GPU_NUMS \
    --lr $LR \
    --worker 32 \
    --test-only \
    --pretrained
