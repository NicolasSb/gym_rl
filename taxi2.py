import gym
import numpy as np

import random
from IPython.display import clear_output

env = gym.make('Taxi-v2').env # try for different environements
print(env)
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.01

iterations = 100

# For plotting metrics
all_epochs = [iterations]
all_penalties = [iterations]
all_score = [iterations]
state = env.reset()


for i in range(iterations):
	epochs, penalties, reward, score = 0, 0, 0, 0
	done = False
	while not done:
		if(random.uniform(0,1) < epsilon):
			action = env.action_space.sample() #explore
		else:
			action = np.argmax(Q[state]) #exploit
		
		next_state, reward, done, info = env.step(action)
		old_val = Q[state, action]
		next_max = np.max(Q[next_state])
		
		new_val = (1-alpha) * old_val + alpha *(reward + gamma * next_max)
		Q[state, action] = new_val
		
		if (reward == -10): 
			penalties += 1
		score += reward
		state = next_state
		epochs += 1
		#env.render()
	print("Score " + str(score))
	print("Penalties " + str(penalties))

	
	if (i%100 == 0):
		clear_output(wait=True)
		print("Episode:" + str(i))
print("Training finished \n")

print(Q)

env.close()
