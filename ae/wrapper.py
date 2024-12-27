import os
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import cv2

from ae.autoencoder import load_ae

import logging

logging.basicConfig(filename='logs/gyro_data.log', level=logging.INFO)


class AutoencoderWrapper(gym.Wrapper):
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param ae_path: Path to the autoencoder
    """

    def __init__(self, env: gym.Env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):  # noqa: B008
        super().__init__(env)
        assert ae_path is not None, "No path to autoencoder was provided"
        self.ae = load_ae(ae_path)
        # Update observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size ,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        obs=self.env.reset()
        # Important: Convert to BGR to match OpenCV convention

        #JETRACER
        # encoded_image = self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1])
        # new_obs = np.concatenate([encoded_image.flatten(), [0.0]])

        new_obs = np.concatenate([obs.flatten(), [0.0, 0.0]])
        return new_obs.flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        encoded_image = self.ae.encode_from_raw_image(obs[:,:,::-1])

        speed = infos["speed"]
        steer = infos["steering"]
        print("========== speed, steer=",speed, ",", steer)
        new_obs = np.concatenate([encoded_image.flatten(), [speed, steer]])
        return new_obs.flatten(), reward, done, infos
