# QRSAC
Implementation of Quantile Regression Soft Actor Critic (QSAC).
This repository is based on [RLkit](https://github.com/vitchyr/rlkit) and [DSAC][https://github.com/xtma/dsac], two popular reinforcement learning frameworks implemented in PyTorch.
The core algorithm of QRSAC is in `rlkit/torch/qrsac/`

## Requirements
- python 3.10+

## Usage
You can write your experiment settings in configs/your_config.yaml and run with 
```
python qrsac.py --config your_config.yaml --gpu 0 --seed 0
```
Set `--gpu -1`, your program will run on CPU.



