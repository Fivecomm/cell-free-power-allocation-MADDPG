#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the testing part of the research work:

Guillermo García-Barrios, Manuel Fuentes, David Martín-Sacristán, "A Novel 
MADDPG Algorithm for Efficient Power Allocation in Cell-Free Massive MIMO," 
IEEE Wireless Communications and Networking Conference (WCNC), Milan, Italy, 
2025. [Pending acceptance]

This is version 1.0 (Last edited: 2024-09-17)

@author: Guillermo Garcia-Barrios

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file.
"""

import os
import pathlib
import hdf5storage
import h5py
import numpy as np
from environment_hpo import CellFreeEnv
import torch
from agilerl.algorithms.maddpg import MADDPG
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import trange
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population

##############################################################################
## PARAMETERS TO SET UP

# Data path
DATA_PATH = 'dataset/'
# Cell-free scenario
TYPE_UES = 'movingUEs' # Two options: 'fixedUEs' or 'movingUEs'
SCENARIO = 'scenario_01'
# Power allocation strategy
PA_STRATEGY = 'sumSE'

# Number of files
FILES = 10
# Total time slots
TOTAL_STEPS = 1000

## MADDPG
MODEL_PATH = 'models/MADDPG_' + TYPE_UES + '_' + SCENARIO + '_hpo.pt'

# Saving path
RESULTS_PATH = 'results/'

max_steps = 10 # time slots per episode

##############################################################################
## LOAD DATA

steps_per_file = TOTAL_STEPS // FILES

# Load scenario data
mat_contents = hdf5storage.loadmat(DATA_PATH + TYPE_UES + '_' + SCENARIO + 
                                   '/setup.mat')
K = int(mat_contents['K'])      # number of UEs
L = int(mat_contents['L'])      # number of APs
Pmax = mat_contents['rho_tot']  # max downlink tx power for each AP
preLogFactor = mat_contents['preLogFactor']

# Variables to calculate SE
bk = np.zeros((L,K,TOTAL_STEPS))     # Avg. ch. gain of desired signal
Ck = np.zeros((L,L,K,K,TOTAL_STEPS)) # Avg. ch. gains interfering signals
# Matrix where (l,k,n) is one if AP l serves UE k in setup n and zero otherwise
D = np.zeros((L,K,TOTAL_STEPS))
betas = np.zeros((TOTAL_STEPS,K,L))  # Large-scale fading coefficients
SE = np.zeros((K,TOTAL_STEPS)) 
rhokl = np.zeros((TOTAL_STEPS,K,L))

# Join all setups data in one file
for file in range(FILES):
    
    file_number = '%0*d' % (2, file + 1)
    interval = np.arange(file * steps_per_file, (file + 1) * steps_per_file)   
    
    with h5py.File(DATA_PATH + TYPE_UES + '_' + SCENARIO + '/test_part' + 
                   file_number + '.mat', 'r') as f:         
        data = f.get('bk_dist') 
        bk[:,:,interval] = np.transpose(np.array(data))
        data = f.get('ck_dist') 
        Ck[:,:,:,:,interval] = np.transpose(np.array(data))
        data = f.get('D_dist') 
        D[:,:,interval] = np.transpose(np.array(data))
        data = f.get('gainOverNoise') 
        betas[interval,:,:] = np.array(data)
        data = f.get('SE_DL_LPMMSE_' + PA_STRATEGY) 
        SE[:,interval] = np.transpose(np.array(data))
        data = f.get('rhokl_DL_LPMMSE_' + PA_STRATEGY) 
        rhokl[interval,:,:] = np.array(data)

betas = np.reshape(betas, (K, L, TOTAL_STEPS))           
rhokl = np.reshape(rhokl, (K, L, TOTAL_STEPS))
    
##############################################################################
## TEST MADDPG

# Number of episodes
episodes = TOTAL_STEPS // max_steps
# Because one time step per episode is used in the reset function
real_steps = episodes * (max_steps - 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure the environment
env = CellFreeEnv(K, L, Pmax, bk, Ck, preLogFactor, D, betas, max_steps)
env.reset()

state_dim = [env.observation_space(agent).shape for agent in env.agents]
one_hot = False

# Action dimension
action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
discrete_actions = False
max_action = [env.action_space(agent).high for agent in env.agents]
min_action = [env.action_space(agent).low for agent in env.agents]  

# Append number of agents and agent IDs to the initial hyperparameter 
# dictionary
n_agents = env.num_agents
agent_ids = env.agents

# Load the saved agent
maddpg = MADDPG.load(MODEL_PATH, device)

# Restart environment
env.close()
env = CellFreeEnv(K, L, Pmax, bk, Ck, preLogFactor, D, betas, max_steps)

# Locate index of rhokl and SE in the state
idx_se = np.arange(env.beta_dim, env.beta_dim + env.se_dim)

rewards = []  # List to collect total episodic reward
indi_agent_rewards = {
    agent_id: [] for agent_id in agent_ids
}  # Dictionary to collect inidivdual agent rewards

# Main loop
timestep = 0
# Define variables to save SE and rhokl estimations
SE_maddpg = np.zeros((K,real_steps))
rhokl_maddpg = np.zeros((K,L,real_steps)) 
for ep in range(episodes):
    
    state, info = env.reset()
    agent_reward = {agent_id: 0 for agent_id in agent_ids}
    score = 0
    
    for _ in range(max_steps-1):
        
        agent_mask = \
            info["agent_mask"] if "agent_mask" in info.keys() else None
        env_defined_actions = (
            info["env_defined_actions"]
            if "env_defined_actions" in info.keys()
            else None
        )
        
        # Get next action from agent
        cont_actions, discrete_action = maddpg.get_action(
            state,
            training=False,
            agent_mask=agent_mask,
            env_defined_actions=env_defined_actions,
        )
        if maddpg.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions
            
        # Take action in environment
        state, reward, termination, truncation, info = env.step(action)
        
        # Save predictions and actions
        SE_maddpg[:,timestep] = state['ap_0'][0,idx_se]     
        for ap in range(L):
            rhokl_maddpg[:,ap,timestep] = action[agent_ids[ap]]
            
        timestep += 1
        
        # Save agent's reward for this step in this episode
        for agent_id, r in reward.items():
            agent_reward[agent_id] += r
            
        # Determine total score for the episode and then append to rewards list
        score = sum(agent_reward.values())
        
        # Stop episode if any agents have terminated
        if any(truncation.values()) or any(termination.values()):
            break
        
    rewards.append(score)
    
    # Record agent specific episodic reward for each agent
    for agent_id in agent_ids:
        indi_agent_rewards[agent_id].append(agent_reward[agent_id])
    
    print("-" * 15, f"Episode: {ep}", "-" * 15)
    print("Episodic Reward: ", rewards[-1])
    '''
    for agent_id, reward_list in indi_agent_rewards.items():
        print(f"{agent_id} reward: {reward_list[-1]}")
    '''

env.close()

##############################################################################
## SAVE DATA
results_path = RESULTS_PATH + TYPE_UES + '/' + SCENARIO + '/'

# Create directory it doesn't exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

np.save(results_path + 'SE_' + PA_STRATEGY + '_test.npy', SE)
np.save(results_path + 'rhokl_' + PA_STRATEGY + '_test.npy', rhokl)
np.save(results_path + 'SE_' + PA_STRATEGY + '_predicted.npy', SE_maddpg)
np.save(results_path + 'rhokl_' + PA_STRATEGY + '_predicted.npy', rhokl_maddpg)