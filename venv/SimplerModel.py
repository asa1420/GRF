import os.path
import gfootball.env as football_env
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import time
import os
start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to stop tensorflow output messages in red.
gamma = 0.99
lambda_ = 0.95

def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambda_ * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # the 1e-10 residue is to make sure not to divide by zero

def get_model_actor_image(input_dims):
    state_input = Input(shape=input_dims)

    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    # Define model
    model = Model(inputs=[state_input], outputs=[out_actions]) # the model takes as input the current state and outputs a list of actions
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model
def get_model_actor_simple(input_dims):
    state_input = Input(shape=input_dims)
    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input) # second layer? it is a hidden layer with 512 neurons
    x = Dense(256, activation='relu', name='fc2')(x) # Third layer?
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x) # output layer
    # Define model
    model = Model(inputs=[state_input], outputs=[out_actions]) # the model takes as input the current state and outputs a list of actions
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model

def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh')(x) # output node from the neural net is 1 since it is only the q_value

    # Define model
    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model
def get_model_critic_simple(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model

env = football_env.create_environment(env_name="academy_empty_goal", rewards='scoring,checkpoints', representation="simple115v2", render=False)
state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n
tensor_board = TensorBoard(log_dir='./logs')

ppo_steps = 128
target_reached = False
iters = 0
max_iters = 1

#model_actor = get_model_actor_simple(input_dims=state_dims)
#model_critic = get_model_critic_simple(input_dims=state_dims)
model_actor = load_model('models/Empty Goal MSE/model_actor_780_0.10000000149011612.hdf5', custom_objects={'loss': 'categorical_hinge'})
model_critic = load_model('models/Empty Goal MSE/model_actor_780_0.10000000149011612.hdf5', custom_objects={'loss': 'categorical_hinge'})
while iters < max_iters:
    iter_rewards = 0
    states = []
    actions = []
    values = []
    masks = []  # checks if the match is in completed/over state, in which case we want to restart the game
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    for itr in range(ppo_steps): # collect 128 interactions with the game
        state_input = K.expand_dims(state, 0)
        s1 = time.time()
        # action_dist_tensor = model_actor([state_input]) # uses the actor model to predict the best actions
        # action_dist = action_dist_tensor.numpy()
        action_dist = model_actor.predict([state_input])
        print("predict action: " + str(time.time() - s1))
        s2 = time.time()
        # q_value_tensor = model_critic([state_input])
        # q_value = q_value_tensor.numpy()[0, 0]
        q_value = model_critic.predict([state_input])[0, 0]
        print("predict q value: " + str(time.time() - s2))
        action = np.random.choice(n_actions, p=action_dist[0, :]) # picks an action based on the action distribution from the actor model
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1
        observation, reward, done, info = env.step(action)
        iter_rewards = iter_rewards + reward
        #print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
        mask = not done

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)

        state = observation # changing the state variable into the new observation or the next state, otherwise we use the same initial state as input to out models.

        if done:
            env.reset() # reset if the game is done
    state_input = K.expand_dims(state, 0)
    # q_value_tensor = model_critic([state_input])
    # q_value = q_value_tensor.numpy()[0, 0]
    q_value = model_critic.predict([state_input])[0, 0]
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards) # useless without PPO
    states = np.reshape(states, newshape=(ppo_steps, 115))
    actions_probs = np.reshape(actions_probs, newshape=(ppo_steps, n_actions))
    advantages = np.reshape(advantages, newshape=(ppo_steps, 1, 1))
    rewards = np.reshape(rewards, newshape=(ppo_steps, 1))
    values = np.reshape(values, newshape=(ppo_steps+1, 1))[:-1]
    returns = np.reshape(returns, newshape=(ppo_steps, 1))
    actions_onehot = np.reshape(actions_onehot, newshape=(ppo_steps, 1, n_actions))
    actor_loss = model_actor.fit(
            [states], [actions_onehot], verbose=True, shuffle=True, epochs=8,
            callbacks=[tensor_board])
    critic_loss = model_critic.fit([states], [returns], shuffle=True, epochs=8, verbose=True, callbacks=[tensor_board])
    #avg_reward = np.mean([test_reward() for _ in range(5)])
    print('total rewards =' + str(iter_rewards))
    model_actor.save('models/Empty Goal MSE exp/model_actor_{}_{}.hdf5'.format(iters, iter_rewards))
    model_critic.save('models/Empty Goal MSE exp/model_critic_{}_{}.hdf5'.format(iters, iter_rewards))
    iters += 1
    env.reset()
print("time taken to finish whole training: " + str(time.time() - start)) # prints at what time the code ends
env.close()