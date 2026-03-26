#!/bin/bash

# 1. 첫 번째 실험 실행 (Linear Mode)
echo "Starting First Experiment: Mode=linear"
CUDA_VISIBLE_DEVICES=0 python main_smd.py \
    DATA.NAME SMD \
    ORACLEAD.CAUSAL_ENCODER.MODE linear \
    ORACLEAD.DECODER.CAUSAL_RESIDUAL False

# 첫 번째 작업이 끝난 후 잠시 대기 (선택 사항)
sleep 2

# 2. 두 번째 실험 실행 (Top-K Mode)
echo "Starting Second Experiment: Mode=topk"
CUDA_VISIBLE_DEVICES=0 python main_smd.py \
    DATA.NAME SMD \
    ORACLEAD.CAUSAL_ENCODER.MODE topk \
    ORACLEAD.DECODER.CAUSAL_RESIDUAL True

echo "All experiments completed."