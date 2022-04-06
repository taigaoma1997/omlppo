# OML-PPO

RL course project for the PyTorch implementation for OML-PPO (https://iopscience.iop.org/article/10.1088/2632-2153/abc327) without spinningup

## Prerequisites
- python=3.7.4
- torch=1.3.1
- torchvision=0.5.0
- tmm=0.1.7

## Make environments
- make environment: conda create --name py37 python=3.7
- make RLMultilayer package: pip install -e .
- Install any necessary packages. 

## Run experiments

Max length = 6:  
python ppo_absorber_visnir.py --cpu 16 --maxlen 6 --exp_name absorber6 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v0

Max length = 15:  
python ppo_absorber_visnir.py --cpu 16 --maxlen 15 --exp_name perfect_absorber15 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v1

