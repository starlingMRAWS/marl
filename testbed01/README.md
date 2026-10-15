# MARL Pipeline

**Multi-Agent Proximal Policy Optimization (MAPPO)** and Centralized Training with Decentralized Execution (CTDE). The environment integrates **pysocialforce** for realistic pedestrian crowd movement and **Pygame** for real-time visualization.

---

##  Installation & Setup

Follow these steps to set up your local development environment and run the project.

### 1. Create and Activate a Virtual Environment
Open your terminal and run the following commands to create a virtual environment: <br>
Setup miniconda:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
create venv:
```bash
conda create -n swarm_env python=3.10 -y
conda activate swarm_env 
```

### 2. Install Dependencies 
Install all required deep learning, simulation, and rendering libraries inside your active conda environment:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install numpy pygame gymnasium pettingzoo pysocialforce matplotlib
```

### 3. File Structure
The project is structured in a modular fashion to separate environment physics, testing, training, and evaluation:

```plaintext
testbed01/
│
├── checkpoints/             
│   ├── best_actor.pth
│   └── training_performance.png
│
├── drone_crowd_env.py       
├── test_env.py              
├── train_mappo.py           
└── validate_mappo.py        
```

### 4. File Overview
```diff
+ drone_crowd_env.py 
```
(2D Simulation Environment)
- Implements a PettingZoo ParallelEnv standard.
- Uses pysocialforce to drive realistic crowd grouping and migration patterns.
```diff
+ test_env.py
```
(Verification Script)
- A lightweight testing utility used to run random or manual actions inside the environment
```diff
+ train_mappo.py
```
(The Training Pipeline)
- Implements the Centralized Training with Decentralized Execution (CTDE) paradigm
- Centralized Critic: Consumes the global state vector $S$ (all agent positions and pedestrian locations) to estimate value baselines V(S)
- Decentralized Actor: Consumes localized $k$-Nearest Neighbor observations $o_i$ to output continuous action velocity vectors
- Includes stability features: GAE (Generalized Advantage Estimation), Layer Normalization, PPO mini-batching, KL-divergence monitoring, and entropy regularization
```diff
+ validate_mappo.py
```
(Evaluation Harness)
- Loads the best-trained actor weights from checkpoints/best_actor.pth
- Executes deterministic policy actions (evaluating the mean $\mu$ of the Gaussian distribution with zero noise)
- Renders the evaluation episodes live via Pygame and reports performance statistics
