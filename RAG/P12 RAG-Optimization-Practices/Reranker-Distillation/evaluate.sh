#!/bin/bash

# 评估模型在测试集上的性能
python evaluation.py \
  --model_before_path /mnt/workspace/modelscope/BAAI/bge-reranker-v2-m3 \
  --model_after_path ../data/reranker_output/checkpoint-1218 \
  --dataset_path ../data/test.jsonl