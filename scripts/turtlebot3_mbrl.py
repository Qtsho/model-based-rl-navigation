#!/usr/bin/env python

# Authors: Tien Tran, adapted  and cs285 RL course
# mail: quang.tran@fh-dortmund.de

from scripts import *
from helpers.utils import * 
from agent.agent import ReinforceAgent


if __name__ == '__main__':
    rospy.init_node('turtlebot3_mbrl')
    all_logs = []

    """Parameters"""
    n_iter=  1
    num_agent_train_steps_per_iter= 100 #1000
    train_batch_size = 512 ##steps used per gradient step (used for training) 512
    action_size = 2 
    observation_size = 2 # heading, current distance

    """Env, agent objects initialization"""
    env = Env(action_size) 
    agent = ReinforceAgent(env ,action_size, observation_size)
 
    #Training loop:
    #collect paths using policy for learning   
   
    for itr in tqdm(range(n_iter)):
        if itr % 1 == 0:
            print("\n\n********** Iteration %i ************"%itr)
        use_batchsize = 80
        if itr==0:
            use_batchsize = 200 #(random) steps collected on 1st iteration (put into replay buffer) 20000
        #TODO: store training trajectories in pickle file: Pkl


        paths, envsteps_this_batch = sample_trajectories(env, agent.actor, min_timestep_perbatch = use_batchsize , max_path_length= 200)
        agent.add_to_replay_buffer(paths, add_sl_noise= True)
        
        ###START TRAINING 
        env.pause()
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
        env.unpause()

    #TODO: validation
    fig = plt.figure()
    print ("Collect data to validate...")
    action_sequence = agent.actor.sample_action_sequences(num_sequences=1, horizon=5) 
    action_sequence = action_sequence[0]
    print(action_sequence)
    mpe, true_states, pred_states = calculate_mean_prediction_error(env, action_sequence, agent.dyn_models, agent.actor.data_statistics)
    for i in range(agent.dyn_models[0].ob_dim):
        plt.subplot(agent.dyn_models[0].ob_dim/2, 2, i+1)
        plt.plot(true_states["observation"][i], 'g')
        plt.plot(pred_states[:,i], 'r')
        
    fig.suptitle('MPE: ' + str(mpe))
    fig.show()
    fig.savefig(agent.figPATH+'/itr_'+str(itr)+'_predictions.png', dpi=200, bbox_inches='tight')
        
        
        
    
    print(all_logs)
    csvRow = all_logs
    with open(agent.resultPATH, 'a') as f:
           # create the csv writer
           writer = csv.writer(f)
           # write a row to the csv file
           writer.writerow(csvRow)
    f.close()




    