import gym
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class AtariEnv():
    def __init__(self, environment_name, frame_buffer_size=22500):
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        self.env.reset()
        self.step_number = 0
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.current_score = 0
        self.global_step_counter = 0

    def step(self, action, save_step=True):
        #take 4 steps with the same action
        next_img_array = []
        reward_float = []
        done_bool    = []
        info_dict    = []
        is_done = False

        for i in range(4):
            img_array_step, reward_float_step, done_bool_step, info_dict_step = self.env.step(action)
            lives = info_dict_step['ale.lives']

            if done_bool_step: # or lives == 2:
                is_done = True
                self.frame_buffer[-1].reward_list[-1] = -5.0 #negative reward for death
            next_img_array.append(img_array_step)
            if reward_float_step > 0.0:
                clipped_reward = 1.0
            elif reward_float_step < 0.0:
                clipped_reward = -1.0
            else:
                clipped_reward = 0.0
            reward_float.append(clipped_reward)
            self.current_score += clipped_reward
            done_bool.append(done_bool_step)
            info_dict.append(info_dict_step)

            self.global_step_counter += 1
            if self.global_step_counter % 1 == 0:
                self.env.render() #show even and odd frames

        if not is_done:
            if len(self.frame_buffer) > 0:
                previous_atari_frame = self.frame_buffer[-1]
                img_array_list = previous_atari_frame.next_img_array_list
            else:
                img_array_list = next_img_array

            atari_frame = AtariFrame(img_array_list, next_img_array, self.step_number, reward_float, done_bool, info_dict, action)
            if save_step:
                self.frame_buffer.append(atari_frame)
            self.step_number += 1
            return atari_frame

        return None

    def close(self):
        self.env.close()

    def reset(self):
        self.current_score = 0
        self.env.reset()

    def get_actions_taken(self):
        n_actions = self.env.action_space.n
        actions_taken = np.zeros(n_actions)
        for step in range(len(self.frame_buffer)):
            this_frame = self.frame_buffer[step]
            this_action = this_frame.action_taken
            actions_taken[this_action] += 1
        return actions_taken

    def get_total_score(self):
        total_score = 0
        for frame in self.frame_buffer:
            total_score += frame.reward

        return total_score


class AtariFrame():
    discounted_reward = 0
    action_array = None

    def __init__(self, img_array_list, next_img_array_list, frame_index, reward_list, done_bool_list, info_dict_list, action_taken):
        self.img_array_list = img_array_list
        self.next_img_array_list = next_img_array_list
        self.frame_index = frame_index
        self.reward_list = reward_list
        self.done_bool_list = done_bool_list
        self.info_dict_list = info_dict_list
        self.action_taken = action_taken

    def process_img_array(self, frame_index):
        img = self.img_array_list[frame_index]
        img = img.mean(axis=2)  # to greyscale
        img = img / 256.0  # normalize from 0 to 1.
        return img

    def process_next_img_array(self, frame_index):
        img = self.next_img_array_list[frame_index]
        img = img.mean(axis=2)  # to greyscale
        img = img / 256.0  # normalize from 0 to 1.
        return img

    def show_frame(self, title="Image"):
        plt.figure(figsize=(11, 7))
        plt.subplot(121)
        plt.title(title)
        plt.imshow(self.img_array)
        plt.axis("off")
        plt.show()

    def get_processed_frames(self):
        processed_frames_shape = (len(self.img_array_list), 210, 160)
        frames = np.zeros(processed_frames_shape)
        for i in range(len(self.img_array_list)):
            #print("self.img_array_list[{}].shape: {}".format(i, self.img_array_list[i].shape))
            frames[i] = self.process_img_array(i)
        return frames

    def get_next_processed_frames(self):
        processed_frames_shape = (len(self.next_img_array_list), 210, 160)
        frames = np.zeros(processed_frames_shape)
        for i in range(len(self.next_img_array_list)):
            #print("self.img_array_list[{}].shape: {}".format(i, self.img_array_list[i].shape))
            frames[i] = self.process_next_img_array(i)
        return frames

    def getReward(self):
        return np.sum(self.reward_list)

    def show_processed_frame(self, frame_index=0, title="Processed Image"):
        plt.figure(figsize=(11, 7))
        plt.subplot(121)
        plt.title(title)
        plt.imshow(self.process_img_array(frame_index), cmap="gray")
        plt.axis("off")
        plt.show()

    def isFinalFrame(self):
        for done_bool in self.done_bool_list:
            if done_bool:
                return True
        return False

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
