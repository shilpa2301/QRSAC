import pickle
import torch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import gym
from rlkit.envs import make_env
from ae.autoencoder import load_ae
import numpy as np

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



with open('/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/qrsac_donkey-generated-roads_normal-iqn-neutral_2025_08_12_14_45_30_0000--s-0/itr_30.pkl', 'rb') as f:
    state_dict = torch.load(f)

target_policy = TanhGaussianPolicy(
            obs_dim=32,
            action_dim=2,
            hidden_sizes=[256, 256, 256],#, 256, 256],
            dropout_probability=0.1,
            )

target_policy.load_state_dict(state_dict["trainer/target_policy"])
target_policy.eval()
#donkeycar
ae_path = "/home/smukh039/work/QRSAC/ae/model_pkls/ae-32_1704492161_best.pkl"
ae = load_ae(ae_path)

#shilpa
register(id="donkey-generated-roads-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv")

env = make_env('donkey-generated-roads-v0')
env.seed(100)

obs=env.reset()
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

return_ep=0
step_count=0
done = False
while done== False:
    action = target_policy.get_actions(obs, True)
    # print(f"action = {action}")
    action=action.flatten()
    state, reward, done, info = env.step(action)
    return_ep+=reward
    step_count+=1
    #donkeycar
    # obs = ae.encode_from_raw_image(np.squeeze(state[:, :, ::-1]))
    env.render()

obs=env.reset()
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

print(return_ep, step_count)


