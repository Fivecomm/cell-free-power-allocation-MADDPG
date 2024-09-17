#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the training part of the research work:

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

import torch
import tensorflow as tf
import os
import pathlib
import hdf5storage
import h5py
import numpy as np
from datetime import datetime
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from environment_hpo import CellFreeEnv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import trange
# set_env you need to pass a seed for reproducibility


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
FILES = 200
# Total time slots
TOTAL_STEPS = 20000


##############################################################################
## LOAD DATA

steps_per_file = TOTAL_STEPS // FILES # // forces to be integer

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
betas = np.zeros((TOTAL_STEPS,K,L))  # Large-scale fadinf coefficients
# Matrix where (l,k,n) is one if AP l serves UE k in setup n and zero otherwise
D = np.zeros((L,K,TOTAL_STEPS))  

# Join all setups data in one file
for file in range(FILES):
    
    file_number = '%0*d' % (2, file + 1)
    interval = np.arange(file * steps_per_file, (file + 1) * steps_per_file)  
    
    with h5py.File(DATA_PATH + TYPE_UES + '_' + SCENARIO + '/train_part' + 
                   file_number + '.mat', 'r') as f:    
        data = f.get('bk_dist') 
        bk[:,:,interval] = np.transpose(np.array(data))
        data = f.get('ck_dist') 
        Ck[:,:,:,:,interval] = np.transpose(np.array(data))
        data = f.get('D_dist') 
        D[:,:,interval] = np.transpose(np.array(data))
        data = f.get('gainOverNoise') 
        betas[interval,:,:] = np.array(data)
        
betas = np.reshape(betas, (K, L, TOTAL_STEPS))

##############################################################################
## MADDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the network configuration
NET_CONFIG = {
      'arch': 'mlp',                # Network architecture
      'hidden_size': [128, 64]      # Network hidden size
}

# Define the initial hyperparameters
INIT_HP = {
    "POPULATION_SIZE": 4,
    "ALGO": "MADDPG",  # Algorithm
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "CHANNELS_LAST": False,
    "BATCH_SIZE": 32,  # Batch size
    "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
    "EXPL_NOISE": 0.1,  # Action noise scale
    "MEAN_NOISE": 0.0,  # Mean action noise
    "THETA": 0.15,  # Rate of mean reversion in OU noise
    "DT": 0.01,  # Timestep for OU noise
    "LR_ACTOR": 0.001,  # Actor learning rate
    "LR_CRITIC": 0.001,  # Critic learning rate
    "GAMMA": 0.95,  # Discount factor
    "MEMORY_SIZE": 100000,  # Max memory buffer size
    "LEARN_STEP": 100,  # Learning frequency
    "TAU": 0.01,  # For soft update of target parameters
}

max_steps = 10  # time slots per episode
env = CellFreeEnv(K, L, Pmax, bk, Ck, preLogFactor, D, betas, max_steps)
env.reset()

# State dimension
state_dim = [env.observation_space(agent).shape for agent in env.agents]
# Action dimension
action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
INIT_HP["DISCRETE_ACTIONS"] = False
INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]
# Append number of agents and agent IDs to the initial hyperparameter 
# dictionary
INIT_HP["N_AGENTS"] = env.num_agents
INIT_HP["AGENT_IDS"] = env.agents   

# Create a population ready for evolutionary hyper-parameter optimisation
num_envs = 1
one_hot = False
pop = create_population(
    INIT_HP["ALGO"],    # Algorithm
    state_dim,          # State dimension
    action_dim,         # Action dimension     
    one_hot,            # One-hot encoding
    NET_CONFIG,         # Network configuration
    INIT_HP,            # Initial hyperparameters
    population_size=INIT_HP["POPULATION_SIZE"], # Population size
    num_envs=num_envs,  # Number of vectorized envs
    device=device,
)

# Configure the multi-agent replay buffer
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    INIT_HP["MEMORY_SIZE"], # Max replay buffer size
    field_names=field_names,
    agent_ids=INIT_HP["AGENT_IDS"],
    device=device,
)

# Instantiate a tournament selection object (used for HPO)
tournament = TournamentSelection(
    tournament_size=2,  # Tournament selection size
    elitism=True,       # Elitism in tournament selection
    population_size=INIT_HP["POPULATION_SIZE"],  # Population size
    eval_loop=1,        # Evaluate using last N fitness scores
)

# Instantiate a mutations object (used for HPO)
mutations = Mutations(
    algo=INIT_HP["ALGO"],
    no_mutation=0.2,    # Probability of no mutation
    architecture=0.2,   # Probability of architecture mutation
    new_layer_prob=0.2, # Probability of new layer mutation
    parameters=0.2,     # Probability of parameter mutation
    activation=0,       # Probability of activation function mutation
    rl_hp=0.2,          # Probability of RL hyperparameter mutation
    rl_hp_selection=[
        "lr",
        "learn_step",
        "batch_size",
    ],  # RL hyperparams selected for mutation
    mutation_sd=0.1,  # Mutation strength
    agent_ids=INIT_HP["AGENT_IDS"],
    arch=NET_CONFIG["arch"],
    rand_seed=1,
    device=device,
)

# Define training loop parameters
max_steps = (TOTAL_STEPS // INIT_HP["POPULATION_SIZE"]) // 2 # Max steps
learning_delay = 0  # Steps before starting learning
evo_steps = 625  # Evolution frequency
eval_steps = None  # Evaluation steps per episode - go until done
eval_loop = 1  # Number of evaluation episodes
elite = pop[0]  # Assign a placeholder "elite" agent

env.close()
total_steps = 0 # Stores the actual number of training steps
env = CellFreeEnv(K, L, Pmax, bk, Ck, preLogFactor, D, betas, evo_steps)

# Training loop
print("Training...")
pbar = trange(max_steps, unit="step")
while np.less([agent.steps[-1] for agent in pop], max_steps).all():
    pop_episode_scores = []
    for agent in pop:  # Loop through population
        state, info  = env.reset() # Reset environment at start of episode
        scores = np.zeros(num_envs)
        completed_episode_scores = []
        steps = 1 # Stores the number of steps per populatioin loop
        
        for idx_step in range(evo_steps-1 // num_envs):
            agent_mask = \
                info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )
            
            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                state, True, agent_mask, env_defined_actions
            )
            if agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions
            
            next_state, reward, termination, truncation, info = env.step(
                action
            )  # Act in environment
            
            scores += np.sum(np.array(list(reward.values())).transpose(), 
                             axis=-1)
            total_steps += num_envs
            steps += num_envs
            
            # Save experiences to replay buffer
            memory.save_to_memory(
                state,
                cont_actions,
                reward,
                next_state,
                termination,
                is_vectorised=True,
            )
            
            # Learn according to learning frequency
            # Handle learn steps > num_envs
            if agent.learn_step > num_envs:
                learn_step = agent.learn_step // num_envs
                if (
                    idx_step % learn_step == 0
                    and len(memory) >= agent.batch_size
                    and memory.counter > learning_delay
                ):
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)
            # Handle num_envs>learn step; learn multiple times per step in env
            elif (
                len(memory) >= agent.batch_size and \
                    memory.counter > learning_delay
            ):
                for _ in range(num_envs // agent.learn_step):
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

            state = next_state

            # Calculate scores and reset noise for finished episodes
            reset_noise_indices = []
            term_array = np.array(list(termination.values())).transpose()
            trunc_array = np.array(list(truncation.values())).transpose()
            for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                if np.any(d) or np.any(t):
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0
                    reset_noise_indices.append(idx)
            agent.reset_action_noise(reset_noise_indices)
            
        pbar.update(evo_steps // len(pop))

        agent.steps[-1] += steps
        pop_episode_scores.append(completed_episode_scores)
    
    # Evaluate population
    fitnesses = [
        agent.test(
            env,
            swap_channels=INIT_HP["CHANNELS_LAST"],
            max_steps=eval_steps,
            loop=eval_loop,
        )
        for agent in pop
    ]
    mean_scores = [
        (
            np.mean(episode_scores)
            if len(episode_scores) > 0
            else "0 completed episodes"
        )
        for episode_scores in pop_episode_scores
    ]

    print(f"--- Global steps {total_steps} ---")
    print(f"Steps {[agent.steps[-1] for agent in pop]}")
    print(f"Scores: {mean_scores}")
    print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
    print(f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) \
        for agent in pop]}'
    )

    # Tournament selection and population mutation
    elite, pop = tournament.select(pop)
    pop = mutations.mutation(pop)

    # Update step counter
    for agent in pop:
        agent.steps.append(agent.steps[-1])

# Save the trained algorithm
path = "./models/"
filename = "MADDPG_" + TYPE_UES + "_" + SCENARIO + "_hpo.pt"
os.makedirs(path, exist_ok=True)
save_path = os.path.join(path, filename)
elite.save_checkpoint(save_path)

pbar.close()
env.close()