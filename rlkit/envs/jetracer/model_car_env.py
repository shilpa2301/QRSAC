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
from ae.autoencoder import load_ae
import os
import pandas as pd
import numpy as np

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

#shilpa JETRACER
#shilpa JETRACER

import threading
from inputs import get_gamepad
import math
class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0 #start ep
        self.X = 0
        self.Y = 0
        self.B = 0 #end ep
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        #shilpa JETRACER
        # self.start_ep = 0
        # self.end_ep = 0
        # self._ep_thread = threading.Thread(target=self._monitor_ep, args=())
        # self._ep_thread.daemon = True
        # self._ep_thread.start()

    #shilpa JETRACER
    # def _monitor_ep(self):
    #     while True:
    #         if self.A == 1:
    #             self.start_ep = 1
    #         if self.B == 1:
    #             self.end_ep = 1
    #         time.sleep(0.1)  # Sleep for a short duration to prevent high CPU usage
    #
    # def read(self):  # return the buttons/triggers that you care about in this methode
    #     x = self.RightJoystickX / XboxController.MAX_JOY_VAL  # normalize between -1 and 1
    #     y = self.LeftJoystickY / XboxController.MAX_JOY_VAL  # normalize between -1 and 1
    #     return [-x, -y, self.A, self.B]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state
                    # elif event.code == 'ABS_X':
                #     self.LeftJoystickX = event.state
                # elif event.code == 'ABS_RY':
                #     self.RightJoystickY = event.state
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state
                # elif event.code == 'ABS_Z':
                #     self.LeftTrigger = event.state
                # elif event.code == 'ABS_RZ':
                #     self.RightTrigger = event.state
                # elif event.code == 'BTN_TL':
                #     self.LeftBumper = event.state
                # elif event.code == 'BTN_TR':
                #     self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                # elif event.code == 'BTN_NORTH':
                #     self.Y = event.state #previously switched with X
                # elif event.code == 'BTN_WEST':
                #     self.X = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                # elif event.code == 'BTN_THUMBL':
                #     self.LeftThumb = event.state
                # elif event.code == 'BTN_THUMBR':
                #     self.RightThumb = event.state
                # elif event.code == 'BTN_SELECT':
                #     self.Back = event.state
                # elif event.code == 'BTN_START':
                #     self.Start = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY1':
                #     self.LeftDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY2':
                #     self.RightDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY3':
                #     self.UpDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY4':
                #     self.DownDPad = event.state


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
    # print("starting connection")

    # conn.settimeout(5)
    # while not data:
    #     try:
    #         data = conn.recv(INT_BYTE_LIMIT)
    #     except socket.error:
    #         print("timedout!")

    data = conn.recv(INT_BYTE_LIMIT)

    # print("connection successful")
    #shilpa JETRACER
    if not data:
        print("No data received from socket connection")
        return None
    # else:
        # print("Data received !!")

    size = int.from_bytes(data, "big")
    # print("size = ", size)
    data = bytearray()
    # print("bytearray done- len =", len(data))
    while len(data) < size:
        # print("in while loop")
        packet = conn.recv(size - len(data))
        # print("connected to packet")
        if not packet:
            #shilpa JETRACER
            print("No packet received from socket connection")

            return None

        # print("packet received")
        data.extend(packet)
        # print("packet appended to data")
    image = np.frombuffer(data, dtype=np.uint8)
    # print("Got image from data")
    logging.info('Image shape ' + str(image.shape))
    # print('image shape', str(image.shape))
    #image = np.reshape(image, (128, 96, 1))
    # print("about to decode")
    image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    # print("Final image received!")
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


class ModelCar(gym.Env, XboxController):
    metadata = {'render.modes': ['human']}

    #shilpa JETRACER
    # ACTION_NAMES: List[str] = ["steer", "throttle"]
    STEER_LIMIT_LEFT: float = -1.0
    STEER_LIMIT_RIGHT: float = 1.0
    THROTTLE_MIN: float = 0.2  # 0.0
    THROTTLE_MAX: float = 0.6
    VAL_PER_PIXEL: int = 255
    STATE_SIZE: int=32

    def __init__(self):
        super(ModelCar, self).__init__()
        self.discrete = False
        #shilpa
        self.use_localization = False   # Enables VR
        self.use_camera = True
        self.use_encoder = True #False
        self.use_action_history = False
        self.action_hist_length = 10
        self.act_interval = 50      # milliseconds

        self.state = None
        #shilpa JETRACER
        # self.rew_fn = lambda x: 0.
        self.steering = 0.0
        self.throttle = 0.0
        self.battery_level = deque(maxlen=20)
        for i in range(20):
            self.battery_level.append(battery_level_start)
        self.step_count = 0
        
        #shilpa JETRACER
        self.in_track = 2

        if self.use_encoder:
            #shilpa JETRACER
            self.observation_space = spaces.Box(-np.inf , np.inf , shape=(1, self.STATE_SIZE), dtype=np.float32)
            # self.observation_space = spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(32,), dtype=float
            # )
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
            #shilpa JETRACER
            self.action_space = spaces.Box(
                low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
                high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]),
                dtype=np.float32,
            )
            # self.action_space = spaces.Box(
            #     np.array([-1., -1.]).astype(np.float32),
            #     np.array([1., 1.]).astype(np.float32),
            # )

        if self.use_encoder:
            # self.encoder = torch.load('~/Shilpa/pipelines/models/ae_models/ae-32_jetracer_model.pkl')
            self.ae = load_ae('/home/pistar/Shilpa/pipelines/models/ae_models/ae-32_jetracer_model.pkl')

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
        # print("image size =", image.shape)
        #batt = receive_battery(self.conn)
        #self.battery_level.append(batt)

        if self.use_camera:
            if self.use_encoder:
                #shilpa JETRACER
                # this_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
                # image = np.expand_dims(this_transform(image), axis=0)
                # image = torch.Tensor(image).cuda()
                # image, _, _ = self.vae.encode(image)
                # image = image.cpu().detach()
                # observed_state = torch.Tensor(np.expand_dims(np.asarray(image.tolist()[0] + [*self.action_hist]), axis=0))
                image = np.stack((image, image, image), axis=-1)
                # image = torch.Tensor(image).cuda()
                observed_state = self.ae.encode_from_raw_image(np.squeeze(image[:, :, ::-1]))
                # observed_state = observed_state.cpu().detach()
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

        #shilpa
        # state = np.concatenate((state, [[act[1]]]), axis=1)
        # print(state, state.shape)
        pose = self.get_pose()
        self.step_count += 1
        return state, pose

    #shilpa JETRACER
    def rew_fn(self, done: bool) -> float:

        #######################--DEMO----------------------------------
        THROTTLE_REWARD_WEIGHT = 0.1
        CRASH_SPEED_WEIGHT = 5
        REWARD_CRASH = -10
        cte_cross_weight = -10
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.throttle - self.THROTTLE_MIN) / (self.THROTTLE_MAX - self.THROTTLE_MIN)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.throttle / self.THROTTLE_MIN)
        lane_keeping_reward = 1 + throttle_reward
        w_1 = 1.0  # 5.0

        total_reward = w_1 * lane_keeping_reward

        return total_reward



    def step(self, action):
        terminated = False
        truncated = False

        steering, throttle = self.convert_action(action)

        if not self.use_localization:
            #shilpa JETRACER
            if self.B == 1:
                print("B detected")
                self.in_track = 0
                self.A = 0
                self.do_action([0., 0.])
                # in_bounds, in_track = True, True
        #shilpa
        # else:
        #     in_track = self.vr.in_track(self.step_count)

        if self.in_track == 0:
            step_reward = -1
            terminated = True
        else:
            self.state, pose = self.do_action((steering, throttle))
            #shilpa JETRACER
            # step_reward = self.rew_fn([self.step_count, steering, throttle])
        step_reward = self.rew_fn(terminated)
            
        if terminated == False:     # Wait to send act until wall-time interval is hit
            offset = self.act_interval - (int(time.time() * 1000) % self.act_interval)
            # time.sleep(offset/1000)

        # #shilpa JETRACER
        ret_state = self.state
        # print("return state from state=", ret_state)
        if terminated == True:

            self.reset()

        #shilpa JETRACER
        return ret_state, step_reward, terminated, {'speed': throttle, 'steer': steering}
        # return self.state, step_reward, terminated, {}  # TODO gymnaisum upgrade truncated,

    def reset(self, **kwargs):
        #shilpa JETRACER
        # if self.state is None:
        if self.state is None :
            print("Waiting for A")
            while self.A == 0:
                pass
            # if self.A==1:
            print("A detected")
            self.in_track = 1
            # self.do_action([0., 0.])
            # send_action(self.conn, (0., 0.))
            self.state = self.get_state()
            self.B == 0
            # self.state = np.concatenate([self.state.flatten(), [0.0, 0.0]])
            # self.state = np.concatenate((self.state, [[0.0]]), axis=1)
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
                print("Waiting for A")
                while self.A == 0:
                    pass
                # if self.A==1:
                print("A detected")
                self.in_track = 1
                # self.state = self.get_state()
                # self.do_action([0., 0.])

        # print("state in reset=", self.state)
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
