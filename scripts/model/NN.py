from scripts import *
from helpers.utils import *


class FFModel(nn.Module):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__() #init NN torch parent class.

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        print ("Using: ", T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        #define forward pass
        self.delta_network = build_mlp(
            input_size= self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        ).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')) 
        
        #define train operations
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        ) # has the param of delta_network
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
        obs_normalized = normalize (obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize (acs_unnormalized, acs_mean,acs_std)
        
        # predicted change in obs
        concatenated_input = T.cat([obs_normalized, acs_normalized], dim=1) # (s,a)
         # send to GPU if it has
        
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

        obs = from_numpy(obs) #reassignment to move toCUDA
        acs = from_numpy(acs)
        data_statistics = {key: from_numpy(value) for key, value in data_statistics.items()} # shift the value to GPU
        
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
        self.optimizer.step()#update the weights

        return to_numpy(loss)

# MPC class, apply action sequences and choose first action of the best action sequences that maximize the reward function.
