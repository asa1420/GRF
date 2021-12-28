import gfootball.env as football_env
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

gamma = 0.99
lambda_ = 0.95

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

    return model

env = football_env.create_environment(env_name="academy_empty_goal", representation="pixels", render=True)
state = env.reset()
state_dims = env.observation_space.shape
print(state_dims)
n_actions = env.action_space.n
print(n_actions)
ppo_steps = 128
states = []
actions = []
values = []
masks = [] # checks if the match is in completed/over state, in which case we want to restart the game
rewards = []
actions_probs = []
actions_onehot = []
model_actor = get_model_actor_image(input_dims=state_dims)
model_critic = get_model_critic_image(input_dims=state_dims)

for itr in range(ppo_steps): # collect 128 interactions with the game
    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input], steps=1) # uses the actor model to predict the best actions
    q_value = model_critic.predict([state_input], steps=1)  # uses the critic model to predict the q value.
    action = np.random.choice(n_actions, p=action_dist[0, :]) # picks an action based on the action distribution from the actor model
    action_onehot = np.zeros(n_actions)
    action_onehot[action] = 1
    observation, reward, done, info = env.step(action)
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
q_value = model_critic.predict([state_input], steps=1)
values.append(q_value)
returns, advantages = get_advantages(values, masks, rewards)

env.close()