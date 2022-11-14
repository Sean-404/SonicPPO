from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from PPO_Model import PPO
import gym
import retro
import torch
import sys
from datetime import datetime
import keyboard

from sonic_util import make_env

now = datetime.now()
date_string = now.strftime("%d_%m_%Y_%H%M")

# Trains a PPO Sonic model from scratch
def train():
    global env
    global model
    #env = retro.make('SonicTheHedgehog-Genesis', zone_choice)
    env = DummyVecEnv([make_env])

    model = PPO(CnnPolicy, env, verbose=1, tensorboard_log="./sonic_ppo_tensorboard/")

    """
    The num of timesteps should be at least 1_000_000
    for a good result but this will take a while to train the model 
    """
    num_steps = int(input("Enter number of timesteps: "))
    
    print("Training model")
    model.learn(total_timesteps=num_steps, log_interval=1)

    model.save("sonic_model_" + date_string + ".pb")
    print("Model saved!")

# Loads and runs a saved model
def load():
    global env
    global model
    #env = retro.make('SonicTheHedgehog-Genesis', zone_choice)
    env = DummyVecEnv([make_env])
    model_name = input("Enter model name: ")
    try:
        model = PPO.load(model_name, env=env, tensorboard_log="./sonic_ppo_tensorboard/")
    except ValueError:
        print("Model cannot be loaded/not found")
        main()
    else:
        print("Model loaded!")

# Main menu for the user
def main():
    print("=== Sonic the Hedgehog PPO Menu ===\n")
    choice = int(input("Train from scratch (1) Play saved model (2) Exit (3): "))
    if choice==1:
        train()
    elif choice==2:
        load()
    elif choice==3:
        quit()
    
    obs = env.reset()

    # Play the AI
    score = 0
    done = False
    
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        score += rewards
        env.render()
        if keyboard.is_pressed("q"):
            done = True

    print("Score: ", score)
    env.close()

if __name__ == "__main__":
    main()
