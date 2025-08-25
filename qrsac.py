import argparse

import torch

import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.qrsac.qrsac import QRSACTrainer
from rlkit.torch.qrsac.networks import QuantileMlp, softmax
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm

from gym.envs.registration import register
# from gym.envs.registration import register

from rlkit.envs.gym_donkeycar.envs.donkey_env import (
    AvcSparkfunEnv,
    CircuitLaunchEnv,
    GeneratedRoadsEnv,
    GeneratedTrackEnv,
    MiniMonacoEnv,
    MountainTrackEnv,
    RoboRacingLeagueTrackEnv,
    ThunderhillTrackEnv,
    WarehouseEnv,
    WarrenTrackEnv,
    WaveshareEnv,
)

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def get_dummy_env(dummy_env):
    return dummy_env
def experiment(variant):

    dummy_env = make_env(variant['env'])
    expl_env = VectorEnv([lambda: get_dummy_env(dummy_env) for _ in range(variant['expl_env_num'])])
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    print(f"obs dim, action dim={obs_dim}, {action_dim}")

    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env=VectorEnv([lambda: get_dummy_env(dummy_env) for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])

    M = variant['layer_size']
    num_quantiles = variant['num_quantiles']

    zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    target_zf1 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    target_zf2 = QuantileMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        num_quantiles=num_quantiles,
        hidden_sizes=[M, M, M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim , #+ action_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M, M],
        dropout_probability = 0.1,
    )

    target_policy = TanhGaussianPolicy(
        obs_dim=obs_dim ,#+ action_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M, M],
        dropout_probability=0.1,
    )

    #shilpa self_certainty reward
    eval_policy = MakeDeterministic(policy)
    # eval_policy = target_policy

    
    # fraction proposal network
    fp = target_fp = None
    if variant['trainer_kwargs'].get('tau_type') == 'fqf':
        fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
        target_fp = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=num_quantiles,
            hidden_sizes=[M // 2, M // 2],
            output_activation=softmax,
        )
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        #shilpa self_certainty reward
        # device=ptu.device,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    trainer = QRSACTrainer(
        env=dummy_env,
        policy=policy,
        target_policy=target_policy,
        zf1=zf1,
        zf2=zf2,
        target_zf1=target_zf1,
        target_zf2=target_zf2,
        fp=fp,
        target_fp=target_fp,
        num_quantiles=num_quantiles,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantile-Regression Soft Actor Critic')
    parser.add_argument('--config', type=str, default="configs/lunarlander.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = 100 #args.seed
    log_prefix = "_".join(["qrsac", variant["env"][:-3].lower(), str(variant["version"])])
    setup_logger(log_prefix, variant=variant, seed=args.seed)
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)
    set_seed(args.seed)

    #shilpa
    register(id="donkey-generated-roads-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv")
    # register(id="donkey-warehouse-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:WarehouseEnv")
    # register(id="donkey-avc-sparkfun-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:AvcSparkfunEnv")
    # register(id="donkey-generated-track-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:GeneratedTrackEnv")
    # register(id="donkey-mountain-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MountainTrackEnv")
    # register(id="donkey-roboracingleague-track-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:RoboRacingLeagueTrackEnv")
    # register(id="donkey-waveshare-v0", entry_point="gym_donkeycar.envs.donkey_env:WaveshareEnv")
    # register(id="donkey-minimonaco-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MiniMonacoEnv")
    # register(id="donkey-warren-track-v0", entry_point="gym_donkeycar.envs.donkey_env:WarrenTrackEnv")
    # register(id="donkey-thunderhill-track-v0", entry_point="gym_donkeycar.envs.donkey_env:ThunderhillTrackEnv")
    # register(id="donkey-circuit-launch-track-v0", entry_point="gym_donkeycar.envs.donkey_env:CircuitLaunchEnv")


    #JETRACER
    # register(id="jetracer_env", entry_point="rlkit.envs.jetracer.model_car_env:ModelCar")
    # ctrl = XboxController()
    # print("TELEOPERATION READY")
    #
    # while True:
    #     _, _, A, _ = ctrl.read()
    #     if A == 1: break
    # while True:
    #
    #     steering, throttle, _, B = ctrl.read()
    #     if B == 1: break
    #     if throttle > 0.7:
    #         throttle = 0.7
    #     # t1 = threading.Thread(target=write_image_to_disk, args=(obs, i, steering, throttle, pose, act_time))
        # t1.start()

    experiment(variant)
