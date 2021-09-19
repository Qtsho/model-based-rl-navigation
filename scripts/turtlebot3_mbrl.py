#!/usr/bin/env python

# Authors: Tien Tran, adapted  and cs285 RL course
# mail: quang.tran@fh-dortmund.de

import rospy
import os
import numpy as np
import time
import sys
import csv
import copy
 
from tqdm import tqdm

from std_msgs.msg import Float32MultiArray
from environment_mbrl import Env
 
from typing import Union, List
from typing_extensions import TypedDict
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPISODES = 3000
device = None


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

#funtion Path that return an object PathDict

def Path(
    obs: List[np.ndarray],
    acs: List[np.ndarray],
    rewards: List[np.ndarray],
    next_obs: List[np.ndarray], 
    terminals: List[bool],
) -> PathDict:

    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}
   
def sample_trajectory(env, policy, max_path_length):
    '''This wont use render mode-> no imgs_obs'''
    print("sample trajectory until crash or reach max path length")
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals  = [], [], [], [], []
    steps = 0
    while True:
        obs.append(ob)
        ac = policy.get_action(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
       
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
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers) # sequential container. 



def init_gpu(use_gpu=True, gpu_id=0):
    global device
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
    return T.from_numpy(*args, **kwargs).float().to(device)

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


'''RL CLASSES
    replay buffer
    FFmodel
    MPC policy
    Reinforce Agent
    main()
'''

class ReplayBuffer(object):
    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = [] # a list of rollouts/path
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False): # T : use in RLagent, add to replay buffer
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(paths)

        if noised: # add noise to the data default false
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:# if there is no obs
            self.obs = observations[-self.max_size:] #check the unocupied mem position index = self.mem_cntr % self.mem_size
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1): #sample past rollouts
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size): # sample random batch from the replay buffer
        #check if all the data is equal obs 
        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0]) [:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals


class FFModel(nn.Module):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__() #init NN torch parent class.

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate

        self.delta_network = build_mlp(
            input_size= self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(device) # send the network to device  allocate to GPU or CPU

        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def save(self, filepath):
        T.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(T.load(filepath))

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        # normalize input data to mean 0, std 1
        obs_normalized = normalize (obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize (acs_unnormalized, acs_mean,acs_std)

        # predicted change in obs
        concatenated_input = T.cat([obs_normalized, acs_normalized], dim=1) # (s,a)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_network(concatenated_input) #delta t+1

        next_obs_pred = obs_unnormalized + unnormalize (delta_pred_normalized, delta_mean, delta_std)

        return next_obs_pred, delta_pred_normalized


    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """

        obs = from_numpy(obs)
        acs = from_numpy(acs)
        data_statistics = {key: from_numpy(value) for key, value in data_statistics.items()}

        # get numpy array of the predicted next-states (s_t+1)
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs. This is similar to self.forward(state...)
        prediction, _ = self( #next_obs_pred
            obs,
            acs,
            data_statistics['obs_mean'],
            data_statistics['obs_std'],
            data_statistics['acs_mean'],
            data_statistics['acs_std'],
            data_statistics['delta_mean'],
            data_statistics['delta_std'],
        )
        return prediction.cpu().detach().numpy()

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """
        
        observations = from_numpy(observations)
        actions = from_numpy(actions)
        next_observations = from_numpy(next_observations)
        data_statistics = {key: from_numpy(value) for key, value in
                           data_statistics.items()}
        target = normalize(
            next_observations - observations,
            data_statistics['delta_mean'],
            data_statistics['delta_std'],
        ) # TODO(Q1) compute the normalized target for the model.
        # Hint: you should use `data_statistics['delta_mean']` and
        # `data_statistics['delta_std']`, which keep track of the mean
        # and standard deviation of the model.


        _, pred_delta = self( #delta_pred_normalized
            observations,
            actions,
            data_statistics['obs_mean'],
            data_statistics['obs_std'],
            data_statistics['acs_mean'],
            data_statistics['acs_std'],
            data_statistics['delta_mean'],
            data_statistics['delta_std'],
        )
        loss = self.loss(target, pred_delta) 
        
        self.optimizer.zero_grad()
        loss.backward() #gradient decent on the loss
        self.optimizer.step()

        return to_numpy(loss)

# MPC class, apply action sequences and choose first action of the best action sequences that maximize the reward function.

class MPCPolicy():

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  

        self.ob_dim = self.env.observation_space[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        # self.low = self.ac_space.low # min action
        # self.high = self.ac_space.high #max action
        self.low = -0.5 # m/s,angular/s
        self.high = 0.5# m/s, angular/s
    
    def sample_action_sequences(self, num_sequences, horizon): 
        #  uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high] # TODO RANDOM Shooting
        random_action_sequences = np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            print("WARNING: performing random actions.",self.sample_action_sequences(num_sequences=1, horizon=1)[0][0]) #(low, high, size), still MPC but random
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0][0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)
        
        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []

        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model) # shape (N,)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        predicted_rewards = np.mean(
            predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N
        
        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[predicted_rewards.argmax()]
        action_to_take = best_action_sequence[0] # first index of the best action sequence
        
        print ('most optimize actions: ', action_to_take )
        return action_to_take  # Unsqueeze the first index? do we need unsqeezing= adding 1 dim

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model: FFModel):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action. T: this can be CEM or random shooting
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered #T: or samples
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pyT in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        N, H, _ = candidate_action_sequences.shape # number of action sequences, horizon, action dims
       
        pred_obs = np.zeros((N, H, self.ob_dim))
       
        pred_obs[:, 0] = np.tile(obs, (N, 1))
        rewards = np.zeros((N, H))

        for t in range(H):
            #this get reward accross all the trajectories in one model 
            rewards_at_t = self.env.getReward(pred_obs[:, t], candidate_action_sequences[:, t])
            
            rewards[:, t] = rewards_at_t # N rewards of N action sequences.
            if t < H - 1: # get prediction until the end of the horizon
                pred_obs[:, t + 1] = model.get_prediction(
                    pred_obs[:, t],
                    candidate_action_sequences[:, t],
                    self.data_statistics,
                )# this calculate rewards/predict the whole next observations across all sequences at time step t. Cool?

        sum_of_rewards = rewards.sum(axis=1) #sum across the horizon
        assert sum_of_rewards.shape == (N,) #1D array with total reward accross multiple actions sequences.
        return sum_of_rewards


class ReinforceAgent():
    def __init__(self, env, action_size, state_size):
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.resultPATH = self.Path.replace('rl_move_base/scripts', 'rl_move_base/scripts/trainning_result/result.csv')
        self.modelPATH = self.Path.replace('rl_move_base/scripts', 'rl_move_base/scripts/trainning_result/models/model_')
        self.ensemble_size = 3
        self.load_episode = 0
        #T: ensemble, create multiple dynamics NN
        self.env = env
        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                action_size,
                state_size,
                n_layers = 2,
                size  = 250, #: dimension of each hidden layer
                learning_rate = 0.00025,
            )
            self.dyn_models.append(model) # T: create dyn models and append object to list

        self.actor = MPCPolicy(
            self.env,
            ac_dim= 2,#(v,w)
            dyn_models =self.dyn_models,
            horizon = 5, #mpc_horizon
            N = 100, #mpc_num_action_sequences
        ) 
        
        self.replay_buffer = ReplayBuffer()
    
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        start = 0
        for model in self.dyn_models:
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_ens variable defined above useful
            finish = start + num_data_per_ens

            observations = ob_no[start:finish] 
            actions = ac_na[start:finish] 
            next_observations = next_ob_no[start:finish] 

            # use datapoints to update one of the dyn_models
            loss = model.update(observations, actions, next_observations, self.data_statistics)
            losses.append(loss)

            start = finish

        avg_loss = np.mean(losses)
        print ('avg Loss between essemble: ',avg_loss)
        return avg_loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False): #~ store_trasition, update statistics

        # add data to replay buffer: path should be a dictionary with keys as ob, ac..
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        #updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size): 
       # NOTE: The size of the batch returned here is sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size * self.ensemble_size)

        

if __name__ == '__main__':
    rospy.init_node('turtlebot3_mbrl')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    action_size = 2 
    observation_size = 2 # heading ,current distance, 24 laser scans

    all_logs = []

    """Parameters"""
    n_iter=  20 #number of (dagger) iterations
    num_agent_train_steps_per_iter= 1000 #1000
    train_batch_size = 512 ##steps used per gradient step (used for training) 512

    """Env, agent objects initialization"""
    env = Env(action_size) 
    agent = ReinforceAgent(env ,action_size, observation_size)
 
    #Training loop, on policy RL:
    #collect paths using policy for learning   
   
    for itr in tqdm(range(n_iter)):
        if itr % 1 == 0:
            print("\n\n********** Iteration %i ************"%itr)
        use_batchsize = 800 #steps collected per train iteration (put into replay buffer) 8000
        if itr==0:
            use_batchsize = 2000 #(random) steps collected on 1st iteration (put into replay buffer) 20000
        #TODO: store training trajectories in pickle file: Pkl


        paths, envsteps_this_batch = sample_trajectories(env, agent.actor, min_timestep_perbatch = use_batchsize , max_path_length= 200)
        #add the paths to the replay buffer with noise
        agent.add_to_replay_buffer(paths, True)
    
        for train_step in tqdm(range(num_agent_train_steps_per_iter)): # train m,
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = agent.sample(train_batch_size)
            train_log = agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            # with open(agent.resultPATH, 'a') as f:
            # # create the csv writer
            #     writer = csv.writer(f)
            #     # write a row to the csv file
            #     writer.writerow(train_log)
            # f.close()
            all_logs.append(train_log) # = # of iterration * # of train steps.
        
        
    
    print(all_logs)
    csvRow = all_logs
    with open(agent.resultPATH, 'a') as f:
           # create the csv writer
           writer = csv.writer(f)
           # write a row to the csv file
           writer.writerow(csvRow)
    f.close()


    #TODO: Log model prediction in csv, plot the loss through time. 
    #TODO: collect data for evaluation
        



    