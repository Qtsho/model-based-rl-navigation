#!/usr/bin/env python

# Authors: Tien Tran,
# mail: quang.tran@fh-dortmund.de

from scripts import *
from helpers.utils import * 
from agent.agent import ReinforceAgent


if __name__ == '__main__':
    rospy.init_node('validation')
    # validation
    action_size = 2 
    observation_size = 3 # x,y yaw
    number_of_validation= 10
    env = Env(action_size) 
    agent = ReinforceAgent(env ,action_size, observation_size)
    print ("Initialize dât statistics...")
    #Generate the data statistic
    # paths, envsteps_this_batch = sample_trajectories(env, agent.actor,  
    #                                         min_timestep_perbatch = 100 , max_path_length= 200)
    # agent.add_to_replay_buffer(paths, add_sl_noise = True)
        
    for validiation in tqdm(range(number_of_validation)):
        fig = plt.figure()
        print ("Collect data to validate...")
        action_sequence = agent.actor.sample_action_sequences(num_sequences=1, horizon=20) 
        action_sequence = action_sequence[0]
        mpe, true_states, pred_states = calculate_mean_prediction_error(env, action_sequence, agent.dyn_models, agent.actor.data_statistics)
        for i in range(agent.dyn_models[0].ob_dim):
            plt.subplot(1, 3, i+1)
            plt.plot(true_states["observation"][:,i], 'g', label='Ground Truth')
            plt.plot(pred_states[:,i], 'r', label='Predicted State')
            plt.xlabel('Horizon')
            plt.ylabel('State')
            plt.legend()
        fig.suptitle('Mean Prediction Error: ' + str(mpe))
        fig.savefig(agent.figPATH+'/validation/itr_'+str(validiation)+'.png', dpi=500, bbox_inches='tight')           
        # Print losses
    