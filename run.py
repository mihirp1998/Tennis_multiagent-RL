import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
# from config import Config
# from network import Actor, Critic
# from memory import ReplayBuffer
# from noise import OUNoise
# from agent import DDPGAgent
from mulagent import MultiAgent
import matplotlib.pyplot as plt
# %matplotlib inline
env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

a = open("file.txt",'w+')

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
print("episode done ",env_info.local_done)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
agent = MultiAgent(state_size=state_size, action_size=action_size, num_agents=len(env_info.agents))
#scores = np.zeros(1)                          # initialize the score (for each agent)
print("hello")
def ddpg(n_episodes=5000, max_t=2000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        #/print("mello")
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        #print("kello")
        state = env_info.vector_observations                 # get the current state (for each agent)
        score = np.zeros(len(env_info.agents))
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                        # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
        avg_score = np.mean(score)
        scores_deque.append(avg_score)
        scores.append(avg_score)
        a.write('\rEpisode {}\tAverage Score: {:.8f}'.format(i_episode, np.mean(scores_deque)))
        a.flush()
        print('\rEpisode {}\tAverage Score: {:.8f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 100 == 0:
            torch.save(agent.ddpg_agents[0].actor_local.state_dict(), 'checkpoint_inter_actor.pth')
            torch.save(agent.ddpg_agents[0].critic_local.state_dict(), 'checkpoint_inter_critic.pth') 
        if np.mean(scores_deque) >= 0.5:
            torch.save(agent.ddpg_agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.ddpg_agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores

scores = ddpg()
fig = plt.figure()
ax = fig.add_subplot(111)
print(scores)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("graph.png")
plt.show()
