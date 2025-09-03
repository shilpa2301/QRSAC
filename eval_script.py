import pickle
import torch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import gym
from rlkit.envs import make_env
from ae.autoencoder import load_ae
import numpy as np

from gym.envs.registration import register
# from gym.envs.registration import register

import numpy as np
import csv
from datetime import datetime

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



with open('/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/qrsac_donkey-generated-roads_normal-iqn-neutral_2025_08_31_16_11_16_0000--s-0_RC1_2/itr_85.pkl', 'rb') as f:
    state_dict = torch.load(f)

target_policy = TanhGaussianPolicy(
            obs_dim=34,
            action_dim=2,
            hidden_sizes=[256, 256, 256, 256, 256],
            dropout_probability=0.1,
            )

target_policy.load_state_dict(state_dict["trainer/target_policy"])
target_policy.eval()
#donkeycar
ae_path = "/home/smukh039/work/QRSAC/ae/model_pkls/icra_generated_roads_ae_model.pkl"
ae = load_ae(ae_path)

#shilpa
register(id="donkey-generated-roads-v0", entry_point="rlkit.envs.gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv")

env = make_env('donkey-generated-roads-v0')
env.seed(100)

obs=env.reset()
print (f"obs shape={obs.shape}")
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

# Set the evaluation time and date once before the loop
eval_time_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Define the CSV file path with eval_time_date in the filename
csv_file_path = f"/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_{eval_time_date}_RC1_2_85.csv"
# Write the header to the CSV file (excluding eval_time_date as a column)
header = ["failed", "cte", "vel", "accel", "action_throttle", "action_steer", "distance"]

return_ep=0
step_count=0
done = False

# Write the header to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

# action = [0.0, 0.0]
while done== False:
    # obs = np.concatenate ((obs, action), axis=1)
    action = target_policy.get_actions(obs, True)
    # print(f"action = {action}")
    action=action.flatten()
    # print(f"action = {action.shape}")
    state, reward, done, info = env.step(action)
    return_ep+=reward
    step_count+=1
    obs = state

    # Extract data from info dictionary
    row_data = [
        info.get('failed', False),
        info.get('cte', 0.0),
        info.get('vel', 0.0),
        info.get('accel', 0.0),
        info.get('action_throttle', 0.0),
        info.get('action_steer', 0.0),
        info.get('distance', 0.0)
    ]
    
    # Append the data to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

    #donkeycar
    # obs = ae.encode_from_raw_image(np.squeeze(state[:, :, ::-1]))
    env.render()

obs=env.reset()
#donkeycar
# obs = ae.encode_from_raw_image(np.squeeze(obs[:, :, ::-1]))

print(return_ep, step_count)


