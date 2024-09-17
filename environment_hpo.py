#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script consists of the environment part of the research work:

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


import gymnasium
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from DownlinkSE import computeDownlinkSE


class CellFreeEnv(ParallelEnv):
    metadata = {
        "name": "cell-free_environment_v0",
    }
    
    def __init__(self, K, L, Pmax, bk, Ck, preLogFactor, D, betas,
                 timeSlotsPerEpisode):
        self.K = K
        self.L = L
        self.Pmax = Pmax
        self.bk = bk
        self.Ck = Ck
        self.preLogFactor = preLogFactor
        self.D = D
        self.betas = betas
        
        self.timeSlot = 0
        self.timeSlotsPerEpisode = timeSlotsPerEpisode
        
        # Define all APs as agents
        self.possible_agents = ["ap_" + str(ap) for ap in range(L)]
        
        # Define observation space
        self.beta_dim = self.K
        self.se_dim = self.K
        self.observation_dim = self.beta_dim + self.se_dim
        self._observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,))
            for agent in self.possible_agents
        }
        
        # Define action space: assign random power coefficient to each AP
        self._action_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(self.K,)) 
            for agent in self.possible_agents}        
    
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing 
    # clock cycles required to get each agent's space. If your spaces change 
    # over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,))  
    
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(self.K,))     
    
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network 
        connections or any other environment data which should not be kept 
        around after the user is no longer using the environment.
        """
        pass
    
    # Think this method as the initilization of the environment    
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """        
        
        self.agents = self.possible_agents
        
        rhokl = np.zeros((self.K,self.L))
        ap = 0
        for agent in self.agents:
            action = self.action_space(agent).sample()
            # Ensure sum of power coefficients per AP is not greater than 1
            total_sum = np.sum(action)
            if total_sum > 1:
                action /= np.sum(action)
            # Ensure all elements are between 0 and 1 after normalization
            action = np.clip(action, 0, 1)
            # Compute rhokl
            rhokl[:,ap] = self.Pmax * action.T
            ap += 1
        
        # Observation = [SE_1, SE_2, ..., SE_k]
        SE = computeDownlinkSE(self.bk[:,:,self.timeSlot],
                                self.Ck[:,:,:,:,self.timeSlot],
                                self.preLogFactor,
                                self.K,
                                self.D[:,:,self.timeSlot],
                                rhokl)
        
        # Observations for each agent
        observations = {}
        betas_l = np.zeros((1,self.K))
        ap = 0        
        for agent in self.agents:
            betas_l[0,:] = self.betas[:,ap,self.timeSlot]
            # Concatenate state variables as it must be an array
            observations[agent] = np.concatenate((betas_l, SE), axis=1)
            ap += 1
        
        self.iterator = 1
        self.timeSlot += 1      
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
        
    def step(self, actions): 
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty 
        # observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Join actions into a power coefficient matrix
        rhokl = np.zeros((self.K,self.L))
        ap = 0
        for agent in self.agents:
            action = actions[agent]
            # Ensure sum of power coefficients per AP is not greater than 1
            total_sum = np.sum(action)
            if total_sum > 1:
                action /= np.sum(action)
            # Ensure all elements are between 0 and 1 after normalization
            action = np.clip(action, 0, 1)
            # Compute rhokl
            rhokl[:,ap] = self.Pmax * np.squeeze(action.T)
            ap += 1
        
        # Observation = [SE_1, SE_2, ..., SE_k]
        SE = computeDownlinkSE(self.bk[:,:,self.timeSlot],
                                self.Ck[:,:,:,:,self.timeSlot],
                                self.preLogFactor,
                                self.K,
                                self.D[:,:,self.timeSlot],
                                rhokl)
        
        # Observations for each agent
        observations = {}
        betas_l = np.zeros((1,self.K))
        ap = 0
        for agent in self.agents:
            betas_l[0,:] = self.betas[:,ap,self.timeSlot]
            # Concatenate state variables as it must be an array
            observations[agent] = np.concatenate((betas_l, SE), axis=1)
            ap += 1
        
        # Rewards (sumSE)
        reward = np.array([np.sum(SE)])
        rewards = {a: reward for a in self.agents} 
        
        # Check termination and truncation conditions
        self.iterator += 1
        if  self.iterator >= self.timeSlotsPerEpisode:
            terminations = {a: np.array([True]) for a in self.agents}
            truncations = {a: np.array([True]) for a in self.agents}      
        else:
            terminations = {a: np.array([False]) for a in self.agents} 
            truncations = {a: np.array([False]) for a in self.agents}  
             
        self.timeSlot += 1 
        
        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos