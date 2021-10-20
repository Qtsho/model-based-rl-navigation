from scripts import *
from helpers.utils import *
from model.NN import FFModel
from policy.mpc_policy import MPCPolicy

class ReinforceAgent():
    def __init__(self, env, action_size, state_size):
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.resultPATH = self.Path.replace('rl_move_base/scripts/agent', 'rl_move_base/scripts/result')
        self.modelPATH = self.Path.replace('rl_move_base/scripts/agent', 'rl_move_base/scripts/result/models/model_')
        self.figPATH = self.Path.replace('rl_move_base/scripts/agent', 'rl_move_base/scripts/result/figures')
        self.ensemble_size = 3
        self.load_episode = 0
        #T: ensemble, create multiple dynamics NN
        self.env = env
        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                action_size,
                state_size,
                n_layers = 3,
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
        #print ('avg Loss between essemble: ',avg_loss)
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


    #funtion Path that return an object PathDict
    def sample(self, batch_size): 
        # NOTE: The size of the batch returned here is sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size * self.ensemble_size)
        


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


