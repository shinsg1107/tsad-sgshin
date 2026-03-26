#!/bin/bash

# 1. 첫 번째 실험 실행 (Linear Mode)
echo "Starting First Experiment: Mode=linear"
CUDA_VISIBLE_DEVICES=2 python main_seed.py \
    DATA.NAME SWaT \
    ORACLEAD.CAUSAL_ENCODER.GRAPH_PATH /home/sgshin/workspace/SGTSAD/data/Causal_graph/SWaT_graph.npy \
    ORACLEAD.CAUSAL_ENCODER.MODE linear \
    ORACLEAD.DECODER.CAUSAL_RESIDUAL True

# 첫 번째 작업이 끝난 후 잠시 대기 (선택 사항)
sleep 2

# 2. 두 번째 실험 실행 (Top-K Mode)
echo "Starting Second Experiment: Mode=topk"
CUDA_VISIBLE_DEVICES=2 python main_seed.py \
    DATA.NAME SWaT \
    ORACLEAD.CAUSAL_ENCODER.MODE topk \
    ORACLEAD.CAUSAL_ENCODER.GRAPH_PATH /home/sgshin/workspace/SGTSAD/data/Causal_graph/SWaT_graph.npy \
    ORACLEAD.DECODER.CAUSAL_RESIDUAL True

echo "All experiments completed."