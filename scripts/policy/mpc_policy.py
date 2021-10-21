import numpy as np

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
        self.low = -0.1 # m/s,angular/s
        self.high = 0.1# m/s, angular/s
    
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
        
        #print ('most optimize actions: ', action_to_take )
        return action_to_take  # Unsqueeze the first index? do we need unsqeezing= adding 1 dim

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
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

