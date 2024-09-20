# A Novel MADDPG Algorithm for Efficient Power Allocation in Cell-Free Massive MIMO

This repository contains the dataset and code to reproduce results of the following conference paper:

Guillermo García-Barrios, Manuel Fuentes, David Martín-Sacristán, "A Novel MADDPG Algorithm for Efficient Power Allocation in Cell-Free Massive MIMO," IEEE Wireless Communications and Networking Conference (WCNC), Milan, Italy, 2025. [Pending acceptance]

## Abstract of the Paper

As we move towards beyond-fifth and sixth-generation wireless networks, new challenges emerge, including the need for higher data rates and lower latency. One promising solution to overcome some of these challenges is cell-free massive MIMO systems. However, a significant drawback of such systems is the high computational cost, particularly in downlink power allocation operations. This paper introduces a novel approach to tackle this issue by leveraging the proven capabilities of reinforcement learning (RL) algorithms. Specifically, we develop a multi-agent deep deterministic policy gradient (MADDPG) algorithm to maximize the sum spectral efficiency (SE) in power allocation schemes. In addition, a novel evolutionary hyperparameter optimization technique has been applied for the first time in this context. The proposed solution not only outperforms existing methods in the literature but also accurately replicates the behavior of the sum SE maximization approach solved as a convex problem. Notably, it achieves a significant computational efficiency improvement by reducing computation time by a factor of 1,000 compared to the convex solution. The results have been validated in properly configured cell-free networks, addressing the limitations of previous studies that did not consider a sufficiently large number of access points relative to user equipment. This work represents a significant advancement in the field, offering a promising and efficient solution for future wireless communication systems.

## Content of Code Package

This code package is structured as follows:

- `DownlinkSE.py`: This script estimates the downlink SE per user equipment (UE).
- `environment_hpo.py`: This script implements the environment of the MADDPG algorithm.
- `test_hpo.py`: This script tests the MADDPG model. The output distributed power and SE values are saved in the `results/` folder.
- `train_hpo.py`: This script trains the MADDPG model and saves it in the `models/` folder.
- `plot_results.py`: This script plots the results of the testing data.

See each file for further documentation.

**NOTE:** The directories mentioned earlier should be created manually.

# Associated dataset

This repository is associated with a dataset that contains the simulation of 4 distinct cell-free massive MIMO scenarios, where each scenario includes 21,000 setups. The simulations take into account three power allocation optimization schemes: max-min Spectral Efficiency (SE) fairness, sum SE maximization, and Fractional Power Control (FPC).

The dataset is available at [https://zenodo.org/records/13772814](https://zenodo.org/records/13772814)

**NOTE:** The downloaded files should be placed in a directory named `dataset/`.

# Acknowledgments

This work is supported by the Spanish ministry of economic affairs and digital transformation and the European Union - NextGenerationEU [UNICO I+D 6G/INSIGNIA] (TSI-064200-2022-006).

# License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
