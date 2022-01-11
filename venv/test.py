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

f = open('/home/abdullah/PycharmProjects/GRF/venv/test data/empty goal scenario', 'w')

models_list = ['48_1.0','53_1.0','61_2.0','64_1.0','67_1.0','71_1.0','72_1.0',
               '76_1.0','80_1.0','82_1.0','84_1.0','86_1.0','87_1.0','89_1.0',
               '91_1.0','92_1.0','93_1.0','95_1.0','97_1.0','102_1.0','104_1.0',
               '106_1.0','112_1.0','113_1.0','115_1.0','117_1.0','122_1.0',
               '125_1.0','127_1.0','128_1.0','129_1.0','132_1.0','133_1.0',
               '136_1.0','138_1.0','139_1.0','141_1.0','147_1.0',]
for model in models_list:
    model_actor = load_model('model_actor_{}.hdf5'.format(model), custom_objects={'loss': 'categorical_hinge'})
    state = env.reset()
    done = False
    total_rewards = 0
    restart_counter = 0
    while restart_counter < 10:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        #action = np.argmax(action_probs)
        action = np.random.choice(n_actions, p=action_probs[0, :])
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        print('action=' + str(action) + ', reward=' + str(reward) + ', total goals =' + str(total_rewards) + ', model = ' + model +', probs=' + str(action_probs))
        state = next_state
        if done:
            state = env.reset()
            restart_counter += 1
    string_to_write = [model, ":", str(total_rewards),'\n', str(action_probs),'\n']
    f.write(model + ": " + str(total_rewards) + '\n' + str(action_probs) + '\n')
f.close()