#!/home/tanyi/miniconda3/envs/vidnet/bin/python
import os


cmd = "/home/tanyi/.conda/envs/vidnet/bin/python -m torch.distributed.launch --master_port 12235 --nproc_per_node=4\
        main.py  somethingv1  RGB --arch resnet --net H2CN --num_segments 8 --gd 20 --lr 0.01 \
        --lr_scheduler step --lr_steps 30 45 55 --epochs 60 --batch-size 8 \
        --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 16 \
        --npb"

os.system(cmd)

