import gfootball.env as football_env
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lambda_ = 0.95

def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print("testing...")
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        actions_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(actions_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward+=reward
        limit+=1
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

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

def get_model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))
    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    # Define model
    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
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
masks = []  # checks if the match is in completed/over state, in which case we want to restart the game
rewards = []
actions_probs = []
actions_onehot = []

model_actor = get_model_actor_image(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic_image(input_dims=state_dims)
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))


for itr in range(ppo_steps): # collect 128 interactions with the game
    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1) # uses the actor model to predict the best actions
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
model_actor.fit(
        [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
        [np.reshape(actions_onehot, newshape=(-1, n_actions))], verbose=True, shuffle=True, epochs=8)
model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8, verbose=True)
env.close()