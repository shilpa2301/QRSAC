# QRSAC
Implementation of Quantile Regression Soft Actor Critic (QSAC) from "Outracing champion Gran Turismo drivers with deep reinforcement learning" by Wurman PR, Barrett S, Kawamoto K, MacGlashan J, Subramanian K, Walsh TJ, Capobianco R, Devlic A, Eckert F, Fuchs F, Gilpin L. [[Paper]](https://www.nature.com/articles/s41586-021-04357-7)

This repository is based on [RLkit](https://github.com/vitchyr/rlkit) and [DSAC](https://github.com/xtma/dsac), two popular reinforcement learning frameworks implemented in PyTorch.
The core algorithm of QRSAC is in `rlkit/torch/qrsac/`

## Requirements
- python 3.10+
- All dependencies are available in requirements.txt and environment.yml

## Usage
You can write your experiment settings in configs/your_config.yaml and run with 
```
python qrsac.py --config your_config.yaml --gpu 0 --seed 0
```
Set `--gpu -1`, your program will run on CPU.

## Experiments
3 different experiments are conducted to validate the working of the QRSAC algorithm - on gym environments, on an advanced environment (donkercar) and on an real-world scaled RC car (Jetracer).

### Experiments on Gym environments

### Experiments on DonkeyCar

<img src='./readme_media/DonkeyCar.gif'>

### Experiments on JetRacer

<img src='./readme_media/Jetracer.gif'>



