#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script computes the downlink SE per UE for the research work:

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

import numpy as np

def computeDownlinkSE(bk,Ck,preLogFactor,K,D,rhokl):
    """
    Computes the downlink spectral efficiency (SE) per user equipment (UE) from
    equation (7.23) in [1].

    Parameters:
    bk (np.array): L x K matrix of avg. ch. gain of desired signal
    Ck (np.array): L x L x K x K of avg. ch. gains of interfering signals
    preLogFactor (float) : Pre-log factor in the SE expression of Theorem 5.2.
    K (int): Total number of UEs in the system.
    D (np.array): L x K matrix where (l,k) is one if AP l serves UE k and zero
    otherwise.
    rhokl (np.array): K x L matrix of distributed power values.

    Returns:
    SE (np.array): Vector of length K of the SE per UE.
    
    References:
    [1] Özlem Tuğfe Demir, Emil Björnson, and Luca Sanguinetti (2021) 
        “Foundations of User-Centric Cell-Free Massive MIMO”, 
        Foundations and Trends in Signal Processing: Vol. 14, No. 3-4,
        pp. 162-472. DOI: 10.1561/2000000109.
    """
    
    import numpy as np

    # Array to store the number of APs serving a specific UE
    La = np.zeros(K)

    # Lists to store AP indices serving and not serving a specific UE
    Serv = [[] for _ in range(K)]
    NoServ = [[] for _ in range(K)]

    # Construct the above arrays and lists
    for k in range(K):
        servingAPs = np.where(D[:, k] == 1)
        NoservingAPs = np.where(D[:, k] == 0)
        
        Serv[k] = np.array(servingAPs).tolist()
        NoServ[k] = np.array(NoservingAPs).tolist()
        
        La[k] = len(servingAPs[0])

    # Compute the concatenated matrix whose block-diagonals are square-roots of
    # the non-zero portions of the matrices \tilde{C}_{ki} and the concatenated
    # vectors whose portions are the non-zero elements of \tilde{b}_k
    La = La.astype(int)
    Ck2 = np.zeros((sum(La), sum(La), K))
    bk2 = np.zeros((sum(La), K))
    for k in range(K):
        bk2[sum(La[:k]):sum(La[:k+1]), k] = bk[:La[k], k]
        for i in range(K):
            Ck2[sum(La[:i]):sum(La[:i+1]), 
                sum(La[:i]):sum(La[:i+1]),k] = Ck[:La[i], :La[i], k, i]

    # Compute rho for non-zero elements
    rhokl = rhokl.T
    rho = np.sqrt(rhokl[D == 1])

    # Compute e_k and d_k
    eee = np.zeros(K)
    ddd = np.zeros(K)
    for k in range(K):
        # Compute the numerator and denominator
        numm = np.dot(bk2[:, k], rho)
        denomm = 1 + np.dot(rho.T, np.dot(Ck2[:, :, k], rho))
        
        # Update e_k and d_k
        eee[k] = 1 - abs(numm)**2 / denomm
        ddd[k] = 1 / eee[k]

    # Compute SEs
    SE = preLogFactor * np.log2(ddd)

    return SE
 