#!/bin/sh

NUM_TRAINING_PLAYER=5
N_ITERATIONS=100000
DISCOUNT=0.95
BATCH_SIZE=128
INITIAL_EPS=1.0
EPS_STEP=0.99995
MINIMUM_EPS=0.1
TARGET_MODEL_REFRESH_INTERVAL=100
EVALUATION_INTERVAL=10000
EVALUATION_NUM_GAMES=100
INITIAL_LR=0.001
LR_STEP=20000
LR_GAMMA=0.5
DEVICE=gpu

for NUM_PLAYERS in $@
do
    MODEL_SAVE_PATH="rl-models/DQN_${NUM_PLAYERS}_players"

    python train.py --players $NUM_PLAYERS \
        --training-players $NUM_TRAINING_PLAYER \
        --iterations $N_ITERATIONS \
        --discount $DISCOUNT \
        --batch-size $BATCH_SIZE \
        --initial-eps $INITIAL_EPS \
        --eps-step $EPS_STEP \
        --minimum-step $MINIMUM_EPS \
        --turn-dependent-eps \
        --target-refresh-interval $TARGET_MODEL_REFRESH_INTERVAL \
        --evaluation-interval $EVALUATION_INTERVAL \
        --evaluation-num-games $EVALUATION_NUM_GAMES \
        --initial-lr $INITIAL_LR \
        --lr-step $LR_STEP \
        --lr-gamma $LR_GAMMA \
        --model-save-path $MODEL_SAVE_PATH \
        --device $DEVICE
done