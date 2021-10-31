import rospy
import os
import numpy as np
import time
import sys
import csv
import copy
 
from tqdm import tqdm

from std_msgs.msg import Float32MultiArray
from env.environment_mbrl import Env
 
from typing import Union, List
from typing_extensions import TypedDict
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import pickle


import matplotlib.pyplot as plt

EPISODES = 3000
freq = 1 #control frequency
"""Utilities classes and functions"""
Activation = Union[str, nn.Module] #runtme support for typehint, for the activation function: can be str or nn.Module
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


#a child class with master class from TypedDict
class PathDict(TypedDict):
    observation: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray
    
def Path(
    obs: List[np.ndarray],
    acs: List[np.ndarray],
    rewards: List[np.ndarray],
    next_obs: List[np.ndarray], 
    terminals: List[bool],
) -> PathDict: #return annotation,

    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):
    model = models[0]
    # true
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    ob = env.reset()
    start = 0
    freq = 0.08 #control frequency
    steps = 0
    for ac in action_sequence:
        end = time.time()
        while (end - start) < freq: # sync the loop to 1s
            end = time.time()
        sync_time = time.time()
        #print('Time between two steps: ', sync_time-start)
        start = time.time()
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    true_states = Path(obs, acs, rewards, next_obs, terminals)
    #Predict state
    ob = np.expand_dims(true_states["observation"][0], 0) #take first state.
    pred_states = []
    #H steps prediction
    for ac in action_sequence:
        if len(pred_states) >= steps:
            break
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)
    # mpe
    mpe = np.mean((pred_states- true_states["observation"])**2)

    return mpe, true_states, pred_states

def sample_trajectory(env, policy, max_path_length):
    print("sample trajectory until crash or reach max path length")
    obs, acs, rewards, next_obs, terminals  = [], [], [], [], []
    ob = env.reset()
    steps = 0
    start = 0
    freq = 0.08 #control frequency
    while True:
        end = time.time()
        while (end - start) < freq: # sync the loop to 1s
            end = time.time()
        sync_time = time.time()
        print('Time between two steps: ', sync_time-start)
        start = time.time()
        #TODO Print missing control frequency
        
        
        obs.append(ob)
        ac = policy.get_action(ob) # this take the most time
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        
        #sync the simulation??deving
        # t1 = time.time()
        # t2=time.time()
        # while (t2 - t1) < 0.1: # sync the sim to 0.1*rtf
        #     t2 = time.time()
        # env.pause()
        
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            print('Done!!')
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, acs, rewards, next_obs, terminals) #a fcn that create a dictionary

#run sample_trajectory() until the min_timestep_batch reach -> save in to paths, a list of PathDict    
def sample_trajectories(env, policy, min_timestep_perbatch, max_path_length):
    
    time_step_this_batch = 0
    paths : List[PathDict] = [] #a empty list contains only PathDict object. Typehint, just to be more clear
    
    while time_step_this_batch < min_timestep_perbatch: # collect until minimum time step reach
        path = sample_trajectory(env, policy, max_path_length) # a path should reach done flag or more than max_path_lenght
        paths.append(path)
        time_step_this_batch += path["observation"].shape[0] #take the  dict key observation, then the size of it first dimension
        print('time step this batch: ',time_step_this_batch)

    return paths, time_step_this_batch

'''Funtions For validation'''

def sample_n_trajectories (env, policy, max_path_length,n_trajectories):
    paths = []
    for _ in n_trajectories:
        paths.append(sample_trajectory(env,policy,max_path_length))

    return paths

""""""

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate nparrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths]) # list comprehension and flatten to 1D array
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
     #convert string to activation module
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
   
    layers = []
    in_size = input_size # a holder for last layers size
    
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
        
    layers.append(nn.Dropout(0.2)) # add dropout layer prevent overfitting.
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    #layers = T.nn.ModuleList(layers)

    return nn.Sequential(*layers) # sequential container. 



def init_gpu(use_gpu=True, gpu_id=0):
    global device #not working, not sure why?
    if T.cuda.is_available() and use_gpu:
        device = T.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = T.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id): 
    T.cuda.set_device(gpu_id)

#from np to tensor and send to device
def from_numpy(*args, **kwargs):
    return T.from_numpy(*args, **kwargs).float().to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

#tensor to numpy
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()



def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
#get the
def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean


def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)
