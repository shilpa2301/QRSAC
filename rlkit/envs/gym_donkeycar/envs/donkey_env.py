"""
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from rlkit.envs.gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from rlkit.envs.gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller

#shilpa
import torch
from ae.wrapper import AutoencoderWrapper
import cv2
from ae.autoencoder import load_ae


logger = logging.getLogger(__name__)


def supply_defaults(conf: Dict[str, Any]) -> None:
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 5.0),
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")
#shilpa IMAGE
# CAMERA_HEIGHT = 120
# CAMERA_WIDTH = 160
# MARGIN_TOP = CAMERA_HEIGHT // 3
# ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]
# IMAGE_WIDTH = 160
# IMAGE_HEIGHT = 120
# N_CHANNELS = 3
# RAW_IMAGE_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, N_CHANNELS)
# INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey

    :param level: name of the level to load
    :param conf: configuration dictionary
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_NAMES: List[str] = ["steer", "throttle"]
    STEER_LIMIT_LEFT: float = -0.5 #-1.0 #-0.5 # 0.5 works till 04-0.6 th , for th 0.8-> 1.0
    STEER_LIMIT_RIGHT: float = 0.5 #1.0 #0.5
    THROTTLE_MIN: float = 0.2 #0.2 #0.2 #0.0
    THROTTLE_MAX: float = 0.6 #0.6#1.0 #0.6
    VAL_PER_PIXEL: int = 255


    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None

        if conf is None:
            conf = {}

        conf["level"] = level

        #shilpa
        conf["throttle_min"] = self.THROTTLE_MIN
        conf["throttle_max"] = self.THROTTLE_MAX

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])  # pytype: disable=key-error

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        self.proc = None
        if "exe_path" in conf:
            self.proc = DonkeyUnityProcess()
            # the unity sim server will bind to the host ip given
            self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

            # wait for simulator to startup and begin listening
            time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(conf=conf)
        # self.viewer.seed = 200

        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]),
            dtype=np.float32,
        )

        # camera sensor data
        #shilpa IMAGE
        # self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)
        # print("img size=", self.viewer.get_sensor_size())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1, 32), dtype=np.float32)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]  # pytype: disable=key-error

        # wait until the car is loaded in the scene
        self.viewer.wait_until_loaded()


        # shilpa
        # load_path = "/home/pistar/Shilpa/pipelines/pipeline2/aae-train-donkeycar/logs/ae-32_1704492161_best.pkl"
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # saved_variables = torch.load(load_path, map_location=device)
        # model = cls(**saved_variables["data"])
        # model.load_state_dict(saved_variables["state_dict"])
        # model.to(device)
        # self.ae = model
        # self.ae= torch.load(model_path, map_location=torch.device('cpu'))

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.viewer.set_episode_over_fn(ep_over_fn)

    #shilpa
    # def seed(self, seed: Optional[int] = None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        #shilpa
        # self.np_random = seed
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()
            # shilpa
            # print("step inside pipeline2 donkeyenv completed")
           #shilpa
            # encoded_image = self.ae.encode_from_raw_image(observation[:, :, ::-1])
        #     new_obs = np.concatenate([encoded_image.flatten(), [0.0]])
        #     new_obs = new_obs.flatten()
        # return new_obs, reward, done, info
            # shilpa IMAGE
            # observation = self.preprocess_image(observation)
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self.viewer.reset()
        observation, reward, done, info = self.viewer.observe()
        time.sleep(1)
        #shilpa
        # encoded_image = self.ae.encode_from_raw_image(observation[:, :, ::-1])
        # new_obs = np.concatenate([encoded_image.flatten(), [0.0]])
        # new_obs = new_obs.flatten()
        # return new_obs
        # shilpa IMAGE
        # observation = self.preprocess_image(observation)
        return observation

    def render(self, mode: str = "human", close: bool = False) -> Optional[np.ndarray]:
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self) -> bool:
        return self.viewer.is_game_over()

    #shilpa IMAGE
    # def preprocess_input(self, x: np.ndarray, mode: str = "rl") -> np.ndarray:
    #     """
    #     Normalize input.
    #
    #     :param x: (RGB image with values between [0, 255])
    #     :param mode: One of "tf" or "rl".
    #         - rl: divide by 255 only (rescale to [0, 1])
    #         - tf: will scale pixels between -1 and 1,
    #             sample-wise.
    #     :return: Scaled input
    #     """
    #     assert x.shape[-1] == 3, f"Color channel must be at the end of the tensor {x.shape}"
    #     # RL mode: divide only by 255
    #     x /= 255.0
    #
    #     if mode == "tf":
    #         x -= 0.5
    #         x *= 2.0
    #     elif mode == "rl":
    #         pass
    #     else:
    #         raise ValueError("Unknown mode for preprocessing")
    #     # Reorder channels
    #     # B x H x W x C -> B x C x H x W
    #     # if len(x.shape) == 4:
    #     #     x = np.transpose(x, (0, 2, 3, 1))
    #     # x = np.transpose(x, (2, 0, 1))
    #
    #     return x
    #
    #
    # def preprocess_image(self, image: np.ndarray, convert_to_rgb: bool = False, normalize: bool = True) -> np.ndarray:
    #     """
    #     Crop, resize and normalize image.
    #     Optionnally it also converts the image from BGR to RGB.
    #
    #     :param image: image (BGR or RGB)
    #     :param convert_to_rgb: whether the conversion to rgb is needed or not
    #     :param normalize: Whether to normalize or not
    #     :return:
    #     """
    #     assert image.shape == RAW_IMAGE_SHAPE, f"{image.shape} != {RAW_IMAGE_SHAPE}"
    #     # Crop
    #     # Region of interest
    #     r = ROI
    #     image = image[int(r[1]): int(r[1] + r[3]), int(r[0]): int(r[0] + r[2])]
    #     im = image
    #     # Hack: resize if needed, better to change conv2d  kernel size / padding
    #     if ROI[2] != INPUT_DIM[1] or ROI[3] != INPUT_DIM[0]:
    #         im = cv2.resize(im, (INPUT_DIM[1], INPUT_DIM[0]), interpolation=cv2.INTER_AREA)
    #     # Convert BGR to RGB
    #     if convert_to_rgb:
    #         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     # Normalize
    #     if normalize:
    #         im = self.preprocess_input(im.astype(np.float32), mode="rl")
    #
    #     return im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class GeneratedRoadsEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
    	#shilpa
    	super(GeneratedRoadsEnv, self).__init__(level="generated_road", *args, **kwargs)
        # super(GeneratedRoadsEnv, self).__init__(level="generated_road", seed=10, *args, **kwargs)


class WarehouseEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WarehouseEnv, self).__init__(level="warehouse", *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(AvcSparkfunEnv, self).__init__(level="sparkfun_avc", *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(GeneratedTrackEnv, self).__init__(level="generated_track", *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(MountainTrackEnv, self).__init__(level="mountain_track", *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(RoboRacingLeagueTrackEnv, self).__init__(level="roboracingleague_1", *args, **kwargs)


class WaveshareEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WaveshareEnv, self).__init__(level="waveshare", *args, **kwargs)


class MiniMonacoEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(MiniMonacoEnv, self).__init__(level="mini_monaco", *args, **kwargs)


class WarrenTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WarrenTrackEnv, self).__init__(level="warren", *args, **kwargs)


class ThunderhillTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(ThunderhillTrackEnv, self).__init__(level="thunderhill", *args, **kwargs)


class CircuitLaunchEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(CircuitLaunchEnv, self).__init__(level="circuit_launch", *args, **kwargs)
