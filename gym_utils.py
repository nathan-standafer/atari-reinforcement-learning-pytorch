import gym
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class AtariEnv():

    def __init__(self, environment_name, reward_shift):
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        self.env.reset()
        self.step_number = 0
        self.frame_buffer = deque()
        self.reward_shift = reward_shift

    def step(self, action):
        self.env.render()
        img_array, reward_float, done_bool, info_dict = self.env.step(action)
        atari_frame = AtariFrame(img_array, self.step_number, reward_float, done_bool, info_dict, action)
        self.frame_buffer.append(atari_frame)
        self.step_number += 1
        return atari_frame

    def close(self):
        self.env.close()

    def get_discounted_rewards(self, discount_rate=0.97):
        discounted_rewards = np.zeros(len(self.frame_buffer))

        cumulative_rewards = 0
        for step in reversed(range(len(self.frame_buffer))):
            this_frame = self.frame_buffer[step]
            cumulative_rewards = this_frame.reward + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        
        reward_mean = discounted_rewards.mean()
        reward_std = discounted_rewards.std()

        if reward_std != 0:
            all_rewards = [(discounted_reward - reward_mean)/reward_std for discounted_reward in discounted_rewards]
        else:
            # no points were granted  its all bad
            print("no points granted. Setting all discounted rewards to -1")
            all_rewards = [-1 for discounted_reward in discounted_rewards]
        
        all_rewards = np.roll(all_rewards, self.reward_shift)

        for (frame, reward) in zip(self.frame_buffer, all_rewards):
            frame.discounted_reward = reward

        return all_rewards

    def get_total_score(self):
        total_score = 0
        for frame in self.frame_buffer:
            total_score += frame.reward

        return total_score


class AtariFrame():
    discounted_reward = 0
    action_array = None

    def __init__(self, img_array, frame_index, reward, done_bool, info_dict, action_taken):
        self.img_array = img_array
        self.frame_index = frame_index
        self.reward = reward
        self.done_bool = done_bool
        self.info_dict = info_dict
        self.action_taken = action_taken

    def process_img_array(self):
        img = self.img_array
        img = img.mean(axis=2) # to greyscale
        img = img / 256.0  # normalize from 0 to 1.
        return img

    def show_frame(self, title="Image"):
        plt.figure(figsize=(11, 7))
        plt.subplot(121)
        plt.title(title)
        plt.imshow(self.img_array) 
        plt.axis("off")
        plt.show() 

    def show_processed_frame(self, title="Processed Image"):
        plt.figure(figsize=(11, 7))
        plt.subplot(121)
        plt.title(title)
        plt.imshow(self.process_img_array(), cmap="gray") 
        plt.axis("off")
        plt.show() 



# environment_name = "Pong-v0"
# atari_env = AtariEnv(environment_name, 0)
# print(atari_env.env.action_space)


# for i in range(500):
#     atari_env.step(3)

# atari_env.close()

# thisFrame = atari_env.frames[444]

# #print(thisFrame.img_array)
# print(thisFrame.frame_index)
# #thisFrame.show_frame()
# print(thisFrame.process_img_array().shape)  #(210, 160)
# #thisFrame.show_processed_frame()

# #thisFrame.reward = 3.8

# print(atari_env.apply_discounted_rewards())
