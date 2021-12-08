import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf
import os.path
from tensorflow.keras.models import Model, load_model

env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)

n_actions = env.action_space.n
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

model_actor = load_model('model_actor.hdf5')
model_critic = load_model('model_critic.hdf5')


state = env.reset()
done = False

while True:
    state_input = K.expand_dims(state, 0)
    action_probs = model_actor.predict(state_input, steps=1)
    action = np.argmax(action_probs)
    next_state, _, done, _ = env.step(action)
    state = next_state
    if done:
        state = env.reset()