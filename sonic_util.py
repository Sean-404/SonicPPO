#https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py

import gym
import retro
import numpy as np
from gym import Wrapper
from gym.spaces import Box
import cv2

cv2.ocl.setUseOpenCL(False)

from baselines.common.atari_wrappers import WarpFrame, FrameStack

"""
Shows all the zones and acts from Sonic the Hedgehog (1991)
for the user to choose to train the agent
"""
def show_zones():
    sonic_zones = ["GreenHillZone.Act1","GreenHillZone.Act2","GreenHillZone.Act3",
          "MarbleZone.Act1","MarbleZone.Act2","MarbleZone.Act3",
          "SpringYardZone.Act1","SpringYardZone.Act2","SpringYardZone.Act3",
          "LabyrinthZone.Act1","LabyrinthZone.Act2","LabyrinthZone.Act3",
          "StarLightZone.Act1","StarLightZone.Act2","StarLightZone.Act3",
          "ScrapBrainZone.Act1","ScrapBrainZone.Act2"]
    print("=== Sonic the Hedgehog 1 Zones and Acts ===")
    print("\n".join(sonic_zones))

    # Validate the user's input
    global zone_choice
    validated = False
    while validated == False:
        zone_choice = str(input("Choose a level to train or type exit: "))
        if zone_choice == "exit":
            main()
        elif zone_choice not in sonic_zones:
            print("Input is not a valid level. Try again.")
            main()
        else:
            validated = True

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    show_zones()
    env = retro.make('SonicTheHedgehog-Genesis', zone_choice)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = PreprocessFrame(env)
    if stack:
        env = FrameStack(env, 4)

    env = CustomReward(env)
    env = AllowBacktracking(env)
    return env

class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    Set frame to grey
    Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

    

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._cur_x = 0
        self._cur_y = 0
        self._cur_score = 0
        self._cur_rings = 0
        self._cur_lives = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._cur_y = 0
        self._cur_score = 0
        self._cur_rings = 0
        self._cur_lives = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # Reward the agent for increasing its' x position in the level
        rew = min(max((info['x'] - self._cur_x), 0), 2)
        self._cur_x = max(info['x'], self._cur_x)
        
        rew += (info["score"] - self._cur_score) / 40
        self._cur_score = info["score"]

        rew += min(max(info["rings"] - self._cur_rings, -2), 2)
        self._cur_rings = info["rings"]

        if info["lives"] < 3:
            return obs, -5, True, info
        
        return obs, rew, done, info

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
