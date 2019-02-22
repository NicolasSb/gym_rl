"""
Utility used by the Network class to actually train.
Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
"""

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import History 

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def create_env():
    env = gym.make('MountainCar-v0').env # try for different environements
    state = env.reset()
    np.random.seed(1)
    env.seed(1)
    nb_actions = env.action_space.n
    return (env, state, nb_actions)

def compile_model(network, nb_actions, input_shape, env):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    
    #prepare data
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        #model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_actions, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    history = History()
    env, state, nb_actions = create_env()
    
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    model = compile_model(network=network, nb_actions=nb_actions, input_shape=(1,), env=env)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    print(model.summary())
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
    dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
    
    h = dqn.test(env, nb_episodes=100, nb_max_episode_steps=1000, visualize=False, verbose=1)
    
    score = h.history['episode_reward']
    env.close()
    return score#score[1] # 1 is accuracy. 0 is loss.
