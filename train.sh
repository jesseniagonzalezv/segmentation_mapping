#!/usr/bin/env bash
cd /home/
python train.py --device-ids 0 --limit 100 --batch-size 4 --lr 0.0001 --n-epochs 10 --jaccard-weight 0.3 --model UNet11