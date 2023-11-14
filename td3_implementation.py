import stat
from turtle import st
import tensorflow as tf
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from velodyne_env import GazeboEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for training = according to paper
num_episodes = 800
max_steps = 500
v_max = 0.5 #m/s
w_max = 1 #rad/s

n_delayed_reward = 10 #steps
parameter_update_delay = 2 #episodes
environment_dim = (10,10) #10 x 10 m
# Gaussian noise added to sensor and action values
# every time environment resets, obstacle locations, starts, and goals change
min_dist_threshold = 1 # meter, distance to goal where we return success
environment_dim = 20 # number of laser readings - UNSPECIFIED BY PAPER
robot_dim = 4 # dimension of state  - In paper it makes it seem like robot dimension is 2, relative distance and heading to local waypoint!

# Actor proposes set of possible actions given state
# Critic: Estimated value function, evaluates actions taken by actor based on given policy
# Set seed for experiment reproducibility
seed = 42
# this is tensorflow but... https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
# Set small epsilon value for stabilizing division
# THis is a pytorch tutorial : https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
eps = np.finfo(np.float32).eps.item()



class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # input is size 23
        # FC800, ReLu, FC600, ReLu, FC2, tanh
        self.actor_stack = nn.Sequential(
            nn.Linear(self.state_size, 800),
            nn.ReLU(),
            nn.Linear(800, 600), 
            nn.ReLU(),
            nn.Linear(600,self.action_size),
            nn.Tanh()
        )

    def forward(self, s):
        # s is input data, state
        s = self.flatten(s)
        action = self.actor_stack(s)
        # (Q) should this return a distribution, like here: https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py
        return action

class Critic(nn.Module):
    # Single Critic network!
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1a = nn.Linear(self.state_size, 800)
        self.layer2a = nn.Linear(800, 600)
        self.layer2b = nn.Linear(self.action_size, 600)

        # Equation (6) - combined layer
        # Unclear about this implementation though. - note for final paper!
        # Combined FC layer is not a common term, so originally thought it looked something like below:
        # OLD ==> self.layer3 = self.layer2a + self.layer2b # Plus bias?
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Linear(600, 1) # output layer
    
    def forward(self, s, a):
        # s = state
        # a = action
        s = self.flatten(s)
        a = self.flatten(a)
        Ls = F.relu(self.layer1a(s))
        F.layer2a(out)
        F.layer2b(a)
        out_s = torch.mm(Ls, self.layer2a.weight.data.t())
        out_a = torch.mm(a, self.layer2a.weight.data.t())
        out = self.layer3(out_s + out_a + self.layer2a.bias.data)
        Q1 = self.layer4(out)

        return Q1

def train_td3():
    env = GazeboEnv("multi_robot_scenario.launch", environment_dim)

    """
    1. Run the agent on the environment to collect training data per episode.
    2. Compute expected return at each time step.
    3. Compute the loss for the combined Actor-Critic model.
    4. Compute gradients and update network parameters.
    Repeat 1-4 until either success criterion or max episodes has been reached.
    """

    state_size = robot_dim + environment_dim
    action_size = 2
    actor = Actor(state_size, action_size).to(device)
    critic1 = Critic(state_size, action_size).to(device)
    critic2 = Critic(state_size, action_size).to(device)

    for i in range(num_episodes):
        state = env.reset()
        a = actor(state)

        # Scaling according to eqn 5
        v = v_max * (a(0) + 1)/2
        w = w_max * a(1)

        action = [v,w]

        #(Q) how to implement parameter delay?
        Q1 = critic1(state, action)
        Q2 = critic2(state, action)
        Q = min(Q1, Q2)


              









"""
create and initialize training environment with action space, environmet dim, state sim
create/initialize TD3 network

for i in episodes:

action = network.get_action(state)

next_state, reward, done, target = env.step(action)
"""