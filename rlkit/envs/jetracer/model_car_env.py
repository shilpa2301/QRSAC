import logging
import functools
import socket
import struct
import atexit
import numpy as np
import cv2
import time
from collections import deque


import gym
from gym import spaces
from torchvision import transforms
import torch

# from config import *
# from localization_vr_index import ValveIndex
#shilpa
SERVER_IP = '192.168.2.159'
PORT = 65432
INT_BYTE_LIMIT = 4
CAMERA_WIDTH = 80      # 1280 natively
CAMERA_HEIGHT = 45     # 720 natively
THROTTLE_CHANGE = 0.1
STEERING_CHANGE = 0.1

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

motor_stop = 1380000
motor_min =  1330000
battery_level_start = 203

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        diff = round(time.time() - start_time, 3)
        logging.debug(f"{func.__name__} time: {str(diff)}")
        return result
    return wrapper


@log
def receive_image(conn):
    data = conn.recv(INT_BYTE_LIMIT)
    size = int.from_bytes(data, "big")
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    image = np.frombuffer(data, dtype=np.uint8)
    logging.info('Image shape ' + str(image.shape))
    print('image shape', str(image.shape))
    #image = np.reshape(image, (128, 96, 1))
    image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    return image


@log
def receive_battery(conn):
    batt_lvl = struct.unpack("!f", conn.recv(INT_BYTE_LIMIT))[0]
    return batt_lvl


@log
def send_action(conn, action):
    for act in action:
        packed = struct.pack("!f", act)
        conn.sendall(packed)


def compute_pwm_value(action, polarity, minv, midv, maxv):
    action = np.clip(action, -1.0, 1.0)
    adjVal = action * polarity
    if adjVal < 0:
        ret_val = midv + adjVal * (midv - minv)
    elif adjVal > 0:
        ret_val = midv + adjVal * (maxv - midv)
    else:
        ret_val = midv
    return int(ret_val)


class ModelCar(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ModelCar, self).__init__()
        self.discrete = False
        #shilpa
        self.use_localization = False   # Enables VR
        self.use_camera = True
        self.use_encoder = False
        self.use_action_history = False
        self.action_hist_length = 10
        self.act_interval = 50      # milliseconds

        self.state = None
        self.rew_fn = lambda x: 0.
        self.steering = 0.0
        self.throttle = 0.0
        self.battery_level = deque(maxlen=20)
        for i in range(20):
            self.battery_level.append(battery_level_start)
        self.step_count = 0

        if self.use_encoder:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(32,), dtype=float
            )
        elif self.use_encoder and self.use_action_history:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(32 + self.action_hist_length,), dtype=float
            )
        elif not self.use_encoder and self.use_camera:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(1, 240, 320), dtype=np.uint8
            )
        #shilpa
        # elif self.use_localization:
        #     self.observation_space = spaces.Box(
        #         low=-np.inf, high=np.inf, shape=(2,), dtype=float
        #     )
        else:
            raise NotImplementedError()

        if self.discrete:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(
                np.array([-1., -1.]).astype(np.float32),
                np.array([1., 1.]).astype(np.float32),
            )

        if self.use_encoder:
            self.encoder = torch.load('/home/pistar/Desktop/JetRacer/dataset/dataset/vae.pth')
        #shilpa
        # if self.use_localization:
        #     self.vr = ValveIndex()
        #     self.rew_fn = self.vr.get_reward

        print('Ready')
        if self.use_action_history:
            self.action_hist = deque(maxlen=self.action_hist_length)
            for _ in range(self.action_hist_length):
                self.action_hist.append(0.0)    # TODO make it make action space

        self.motor_offset = 60000
        self.initial_battery_level = None

        self.conn = None

    def connect_car(self):
        # Connect to car
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                s.bind((SERVER_IP, PORT))
                break
            except OSError:
                logging.warning('Retrying socket bind..')
                time.sleep(1)
        s.listen()
        print('Waiting for client connection...')
        self.conn, addr = s.accept()
        atexit.register(self.conn.close)
        print(f"Connected by {addr}")

    @log
    def get_pose(self):
        if not self.use_localization:
            return [-1] * 5
        #shilpa
        # else:
        #     return self.vr.get_pose(self.step_count)

    # ONLY USE get_state on initial reset(), always use do_action otherwise
    @log
    def get_state(self):
        if self.conn is None: self.connect_car()
        image = receive_image(self.conn)
        #batt = receive_battery(self.conn)
        #self.battery_level.append(batt)

        if self.use_camera:
            if self.use_encoder:
                this_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
                image = np.expand_dims(this_transform(image), axis=0)
                image = torch.Tensor(image).cuda()
                image, _, _ = self.vae.encode(image)
                image = image.cpu().detach()
                observed_state = torch.Tensor(np.expand_dims(np.asarray(image.tolist()[0] + [*self.action_hist]), axis=0))
            else:
                observed_state = image
        else:
            x, y, roll, pitch, yaw = self.get_pose()
            observed_state = np.asarray([self.vr.centerline_distance(self.step_count), roll])
        return observed_state

    @log
    def do_action(self, act):
        if self.use_action_history:
            self.action_hist.append(act[0])

        send_action(self.conn, (act[0], act[1]))
        state = self.get_state()
        pose = self.get_pose()
        self.step_count += 1
        return state, pose

    def step(self, action):
        terminated = False
        truncated = False

        steering, throttle = self.convert_action(action)

        if not self.use_localization:
            in_bounds, in_track = True, True
        #shilpa
        # else:
        #     in_track = self.vr.in_track(self.step_count)

        if not in_track:
            step_reward = -1
            terminated = True
        else:
            self.state, pose = self.do_action((steering, throttle))
            step_reward = self.rew_fn([self.step_count, steering, throttle])
            
        if terminated == False:     # Wait to send act until wall-time interval is hit
            offset = self.act_interval - (int(time.time() * 1000) % self.act_interval)
            # time.sleep(offset/1000)

        return self.state, step_reward, terminated, {}  # TODO gymnaisum upgrade truncated,

    def reset(self, **kwargs):
        if self.state is None:
            self.state = self.get_state()
        else:   # Reset vehicle with static policy to move back inside bounds
            #shilpa
            # if self.use_localization:
            #     reset_step = -5
            #     while not self.vr.in_boundary(reset_step):
            #         steer, throttle = self.vr.reset_policy(reset_step)
            #         self.do_action([steer, throttle])
            #         reset_step -= 1
            #         time.sleep(0.5)
            # else:
                self.do_action([0., 0.])
        return self.state#, {}  # TODO gymnasium upgrade(imitation lib doesnt supp)

    def convert_action(self, action):
        if self.discrete:
            self.throttle = 0.6
            if action == 0:   # turn left more
                if self.steering < 1:
                    self.steering = self.steering + STEERING_CHANGE
            elif action == 1:   # turn right more
                if self.steering > -1:
                    self.steering = self.steering - STEERING_CHANGE
            elif action == 2:
                self.throttle = 0.0
            elif action == 3:
                self.steering = 0.0

            # if action == 0:     # noop
            #     pass
            # elif action == 1:   # increase speed
            #     if self.throttle < 1:
            #         self.throttle = self.throttle + THROTTLE_CHANGE
            # elif action == 2:   # decrease speed
            #     if self.throttle > -1:
            #         self.throttle = self.throttle - THROTTLE_CHANGE
            # elif action == 3:   # turn left more
            #     if self.steering < 1:
            #         self.steering = self.steering + STEERING_CHANGE
            # elif action == 4:   # turn right more
            #     if self.steering > -1:
            #         self.steering = self.steering - STEERING_CHANGE
            # else:
            #     raise NotImplementedError('Action undefined')
        # else:
        #         raise NotImplementedError('Action undefined')
        self.steering = action[0]
        self.throttle = action[1]
        return self.steering, self.throttle

    def render(self, mode='human', close=False):
        pass
