import pickle
import torch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import gym
from rlkit.envs import make_env
# from ae.autoencoder import load_ae
import numpy as np


with open('./eval_models/params.pkl', 'rb') as f:
    state_dict = torch.load(f)

target_policy = TanhGaussianPolicy(
            obs_dim=111,
            action_dim=8,
            hidden_sizes=[256, 256, 256, 256, 256],
            dropout_probability=0.1,
            )

target_policy.load_state_dict(state_dict["trainer/policy"])
#donkeycar
# ae_path = "/home/pipelines/pipeline2/aae-train-donkeycar/logs/ae-32_1704492161_best.pkl"
# ae = load_ae(ae_path)
env = make_env('Ant-v2')
env.seed(0)

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


