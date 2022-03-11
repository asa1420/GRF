import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf
import os.path
from tensorflow.keras.models import Model, load_model

env = football_env.create_environment(env_name='academy_3_vs_1_with_keeper', representation='simple115v2', render=True, rewards='scoring,checkpoints', number_of_left_players_agent_controls=2)

action_dims = env.action_space.nvec # the number of actions now is an array of the number of actions for each player. For two players, it is [19 19]

dummy_n = np.zeros((1, len(action_dims), action_dims[0])) # len(action_dims) = number of players being controlled
dummy_1 = np.zeros((1, len(action_dims), 1))

#model_actor = load_model('models/Empty Goal Models/model_actor_147_1.0.hdf5', custom_objects={'loss': 'categorical_hinge'})
model_actor = load_model('models/3vs1_two_check_5M/model_actor_1600_3.6000002026557922.hdf5', custom_objects={'loss': 'categorical_hinge'})
state = env.reset()
done = False
total_rewards = np.zeros(2)
restart_counter = 0
while True:
    state_input = K.expand_dims(state, 0)
    #action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
    action_dist = model_actor([state_input, dummy_n, dummy_1, dummy_1, dummy_1]).numpy()
    action_player1 = np.random.choice(action_dims[0], p=action_dist[0, 0, :])
    action_player2 = np.random.choice(action_dims[0], p=action_dist[0, 1, :])
    next_state, reward, done, _ = env.step([action_player1, action_player2])
    total_rewards[0] += reward[0]
    total_rewards[1] += reward[1]
    print('action_player1=' + str(action_player1) + ', action_player2=' + str(
        action_player2) + ', reward=' + str(reward) + ', total rewards=' + str(total_rewards))
    state = next_state
    if done:
        state = env.reset()
