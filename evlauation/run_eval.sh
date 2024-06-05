#!/bin/bash

for lang in sw bn te th ja zh ru es fr de en
do
  CUDA_VISIBLE_DEVICES=0 python infer.py \
    --lang ${lang} \
    --model laihuiyuan/mCoT \
    --prompt_path ../data/prompt.txt \
    --inp_path ../data/mgsm/test_${lang}.json \
    --out_path ../outputs/mcot_mgsm_${lang}.json
done


for lang in sw bn th ja zh ru es fr de en
do
  CUDA_VISIBLE_DEVICES=0 python infer.py \
    --lang ${lang} \
    --model laihuiyuan/mCoT \
    --prompt_path ../data/prompt.txt \
    --inp_path ../data/msvamp/test_${lang}.json \
    --out_path ../outputs/mcot_msvamp_${lang}.json
done