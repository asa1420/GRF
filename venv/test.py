import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf
import os.path
from tensorflow.keras.models import Model, load_model

env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115', render=True)

n_actions = env.action_space.n
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

model_actor = load_model('model_actor_8_1.0.hdf5', custom_objects={'loss': 'categorical_hinge'})
#model_critic = load_model('model_critic_3_[[1.]].hdf5')


state = env.reset()
done = False
# state_input = K.expand_dims(state, 0)
# action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
# action = np.random.choice(n_actions, p=action_probs[0, :])
# next_state, reward, done, _ = env.step(action)
# print('action=' + str(action) + ', reward=' + str(reward) + 'probs=' + str(action_probs))
# state = next_state
while True:
    state_input = K.expand_dims(state, 0)
    action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
    action = np.argmax(action_probs)
    next_state, reward, done, _ = env.step(action)
    print('action=' + str(action) + ', reward=' + str(reward) + ', probs=' + str(action_probs))
    state = next_state
    if done:
        state = env.reset()