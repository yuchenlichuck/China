
#!/bin/bash
#SBATCH -N 1
#SBATCH --time=21:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuchen.li@kaust.edu.sa
#SBATCH --mem=60G
#SBATCH --gres=gpu:2
#SBATCH --constraint=[v100,rtx2080ti,gtx1080ti]
#SBATCH -J sustc
#SBATCH -o sustc.%J.out
#SBATCH -e sustc.%J.err

python train.py --experiment_name DiffAugment-biggan-cifar10-0.05 --DiffAugment translation,cutout,color \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 --num_samples 2500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 2000 --seed 0

python train.py --experiment_name DiffAugment-biggan-cifar10-0.05 --DiffAugment cutout,color \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 --num_samples 2500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 2000 --seed 0

python train.py --experiment_name DiffAugment-biggan-cifar10-0.05 --DiffAugment translation,color \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 --num_samples 2500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 2000 --seed 0

