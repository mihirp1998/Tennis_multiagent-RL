[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Multiagent RL

![Trained Agent][image1]

### Information
This project is for multi agent RL, which deals in continous action space to take actions such that it maximizes the score of the agents..So that both they play tennis for longer time


Number of Visual Observations (per agent): 0

Vector Observation space type: continuous

Vector Observation space size (per agent): 8

Number of stacked Vector Observation: 3

Vector Action space type: continuous

Vector Action space size (per agent): 2

Vector Action descriptions: In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

        
### Installation

``` bash
conda install --file python3.yml
```

### File Info

Tennis.app --> Unity enviornment for Mac devices

checkpoint_actor.pth,checkpoint_critic.pth ----> Saved model

ddpg_agent.py --> agent file for learning

model2.py --> models for actor and critic

run.py --> file for training

mulagent.py --> multi agent file for taking actions and steps

temp.ipynb --> Ipython notebok for training





