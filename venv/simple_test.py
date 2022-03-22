import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf
import os.path
from tensorflow.keras.models import Model, load_model
import time

env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115v2', render=True, rewards='scoring,checkpoints', number_of_left_players_agent_controls=1)

n_actions = env.action_space.n
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

model_actor = load_model('models/Empty_Goal/model_actor_700_0.800000011920929.hdf5', custom_objects={'loss': 'categorical_hinge'})

state = env.reset()
done = False
total_rewards = 0
restart_counter = 0
while True:
    state_input = K.expand_dims(state, 0)
    s1 = time.time()
    #action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1) # uses the actor model to predict the best actions
    action_dist = model_actor([state_input, dummy_n, dummy_1, dummy_1, dummy_1]).numpy()
    #action_dist = action_dist_tensor.numpy()
    #print("predict action: " + str(time.time() - s1))
    action = np.random.choice(n_actions, p=action_dist[0, :])
    next_state, reward, done, _ = env.step(action)
    total_rewards += reward
    print('action=' + str(action) + ', reward=' + str(reward) + ', total rewards =' + str(total_rewards) + ', probs =' + str(action_dist))
    state = next_state
    if done:
        state = env.reset()
