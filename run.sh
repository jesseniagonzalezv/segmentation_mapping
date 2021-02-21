#!/bin/ bash
#bash trainHR.sh
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,1,2,3
echo hola

#python train.py   --model UNet  --n-epochs 30 --lr 1e-3 --batch-size 4 
#python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet' --count 716 --n-epochs  30

#python train.py   --model UNet11  --n-epochs 40 --lr 1e-4 --batch-size 4 
#python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'UNet11' --count 361 --n-epochs  40

python train.py   --model DeepLabV3  --n-epochs 40 --lr 1e-4 --batch-size 4 
python plotting.py --out-file '512' --stage 'test' --name-file '_100_percent_512' --name-model 'DeepLabV3' --count 361 --n-epochs  40
