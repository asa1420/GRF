import gfootball.env as football_env
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Layer # Layer for the custom call function
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution # hehe
import time
from PPO_loss_layer import PPO_loss_layer
import os
start = time.time()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to stop tensorflow output messages in red. still does not work.
#disable_eager_execution()
#enable_eager_execution()  # to use .numpy()
clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = np.zeros((ppo_steps, len(action_dims)))
    gae1 = 0
    gae2 = 0
    gae3 = 0
    for i in reversed(range(len(rewards))):
        delta1 = rewards[i][0] + gamma * values[i + 1][0] * masks[i] - values[i][0]
        gae1 = delta1 + gamma * lmbda * masks[i] * gae1
        delta2 = rewards[i][1] + gamma * values[i + 1][1] * masks[i] - values[i][1]
        gae2 = delta2 + gamma * lmbda * masks[i] * gae2
        delta3 = rewards[i][2] + gamma * values[i + 1][2] * masks[i] - values[i][2]
        gae3 = delta3 + gamma * lmbda * masks[i] * gae3
        returns[i][0] = gae1 + values[i][0]
        returns[i][1] = gae2 + values[i][1]
        returns[i][2] = gae3 + values[i][2]

    adv = returns - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # advantages is normalised and a residue is added to prevent division by zero


def ppo_loss_print(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        y_true = tf.Print(y_true, [y_true], 'y_true: ')
        y_pred = tf.Print(y_pred, [y_pred], 'y_pred: ')
        newpolicy_probs = y_pred
        # newpolicy_probs = y_true * y_pred
        newpolicy_probs = tf.Print(newpolicy_probs, [newpolicy_probs], 'new policy probs: ')

        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        ratio = tf.Print(ratio, [ratio], 'ratio: ')
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        actor_loss = tf.Print(actor_loss, [actor_loss], 'actor_loss: ')
        critic_loss = K.mean(K.square(rewards - values))
        critic_loss = tf.Print(critic_loss, [critic_loss], 'critic_loss: ')
        term_a = critic_discount * critic_loss
        term_a = tf.Print(term_a, [term_a], 'term_a: ')
        term_b_2 = K.log(newpolicy_probs + 1e-10)
        term_b_2 = tf.Print(term_b_2, [term_b_2], 'term_b_2: ')
        term_b = entropy_beta * K.mean(-(newpolicy_probs * term_b_2))
        term_b = tf.Print(term_b, [term_b], 'term_b: ')
        total_loss = term_a + actor_loss - term_b
        total_loss = tf.Print(total_loss, [total_loss], 'total_loss: ')
        return total_loss

    return loss


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

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    model.summary()
    return model


def get_model_actor_simple(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(len(action_dims), output_dims[0],)) # changed to make it suitable for two players
    advantages = Input(shape=(len(action_dims), 1,))
    rewards = Input(shape=(len(action_dims), 1,))
    values = Input(shape=(len(action_dims), 1,))
    #loss_layer = PPO_loss_layer(2)(state_input, oldpolicy_probs, advantages, rewards, values)
    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input) # second layer? it is a hidden layer with 512 neurons
    x = Dense(256, activation='relu', name='fc2')(x) # Third layer?
    out_actions = Dense(action_dims[0], activation='softmax', name='predictions')(x) # output layer
    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)], experimental_run_tf_function=False)
    #model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model


def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh')(x)

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

def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot

image_based = False

if image_based:
    env = football_env.create_environment(env_name='academy_3_vs_1_with_keeper', representation='pixels', render=False)
else:
    env = football_env.create_environment(env_name='academy_3_vs_1_with_keeper', representation='simple115v2', render=False, rewards='scoring,checkpoints', number_of_left_players_agent_controls=3)

state = env.reset()
state_dims = env.observation_space.shape
action_dims = env.action_space.nvec # the number of actions now is an array of the number of actions for each player. For two players, it is [19 19]

dummy_n = np.zeros((1, len(action_dims), action_dims[0])) # len(action_dims) = number of players being controlled
dummy_1 = np.zeros((1, len(action_dims), 1)) # the extra 1 just adds an extra [] to the array, idk why it is needed (definitely for the neural network though)
# I think it is the dimension for the batch size!
tensor_board = TensorBoard(log_dir='./logs')

if image_based:
    model_actor = get_model_actor_image(input_dims=state_dims, output_dims=action_dims)
    model_critic = get_model_critic_image(input_dims=state_dims)
else:
    model_actor = get_model_actor_simple(input_dims=state_dims, output_dims=action_dims)
    model_critic = get_model_critic_simple(input_dims=state_dims)
    # model_actor = load_model('third_model_actor.hdf5', custom_objects={'loss': 'categorical_hinge'})
    # model_critic = load_model('third_model_critic.hdf5', custom_objects={'loss': 'categorical_hinge'})

ppo_steps = 256
target_reached = False
best_reward = 0
iters = 0
max_iters = 801

while not target_reached and iters < max_iters:
    iter_rewards = np.zeros(len(action_dims))
    states = []
    actions_player1 = []
    actions_player2 = []
    actions_player3 = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None
    for itr in range(ppo_steps):
        state_input = K.expand_dims(state, 0)
        action_dist_tensor = model_actor([state_input, dummy_n, dummy_1, dummy_1, dummy_1])
        action_dist = action_dist_tensor.numpy()
        #q_values = model_critic([state_input])[0, :, 0]
        q_values_tensor = model_critic([state_input])
        q_values = q_values_tensor.numpy()[0, :, 0]
        # edit action_dist by setting maximum probabilities:
        if max(action_dist) > 0.5:
            diff = max(action_dist) - 0.5
            share = diff / (len(action_dist) - 1) # distribute the difference to other actions
            index = np.argmax(action_dist)  # index of max
            for i in range(len(action_dist)):
                if i == index:
                    action_dist[i] = 0.5
                else:
                    action_dist[i] = action_dist[i] + share
        action_player1 = np.random.choice(action_dims[0], p=action_dist[0, 0, :]) # same thing as action_dist, it just removes the extra dimension from model_actor.predict()
        action_player2 = np.random.choice(action_dims[0], p=action_dist[0, 1, :])
        action_player3 = np.random.choice(action_dims[0], p=action_dist[0, 2, :])
        action_onehot = np.zeros((len(action_dims), action_dims[0]))
        action_onehot[0][action_player1] = 1
        action_onehot[1][action_player2] = 1
        action_onehot[2][action_player3] = 1
        observation, reward, done, info = env.step([action_player1, action_player2, action_player3])
        iter_rewards[0] = iter_rewards[0] + reward[0]
        iter_rewards[1] = iter_rewards[1] + reward[1]
        iter_rewards[2] = iter_rewards[2] + reward[2]
        #print('itr: ' + str(itr) + ', action_player1=' + str(action_player1) + ', action_player2=' + str(action_player2) + ', reward=' + str(reward) + ', q val=' + str(q_values) + ', total rewards=' + str(iter_rewards))
        mask = not done

        states.append(state)
        actions_player1.append(action_player1)
        actions_player2.append(action_player2)
        actions_player3.append(action_player3)
        actions_onehot.append(action_onehot)
        values.append(q_values)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)

        state = observation
        if done:
            env.reset()
    q_values = model_critic([state_input]).numpy()[0, :, 0] # the bracket at the end is to get rid of the redundant first dimension
    values.append(q_values)
    returns, advantages = get_advantages(values, masks, rewards)

    #states = np.reshape(states, newshape=(ppo_steps, 72, 96, 3)) for pixels
    states = np.reshape(states, newshape=(ppo_steps, len(action_dims), 115))
    actions_probs = np.reshape(actions_probs, newshape=(ppo_steps, len(action_dims), action_dims[0]))
    advantages = np.reshape(advantages, newshape=(ppo_steps, len(action_dims), 1))
    rewards = np.reshape(rewards, newshape=(ppo_steps, len(action_dims), 1))
    values = np.reshape(values, newshape=(ppo_steps+1, len(action_dims), 1))[:-1]
    returns = np.reshape(returns, newshape=(ppo_steps, len(action_dims), 1))
    actions_onehot = np.reshape(actions_onehot, newshape=(ppo_steps, len(action_dims), action_dims[0]))

    critic_loss = model_critic.fit([states], [returns], shuffle=True, epochs=1,
                                   verbose=True, callbacks=[tensor_board])

    actor_loss = model_actor.fit(
        [states, actions_probs, advantages, rewards, values],
        [actions_onehot], verbose=True, shuffle=True, epochs=1,
        callbacks=[tensor_board])
    print('total test reward of iteration {} = {}'.format(iters, iter_rewards[0]))
    #print('total rewards player 1=' + str(iter_rewards[0]) + 'total rewards player 2=' + str(iter_rewards[1]))
    if not iters % 200:  # save actor models in increments of 200
        model_actor.save('models/3vs1_three_check_max/model_actor_{}_{}.hdf5'.format(iters, iter_rewards[0]))
        model_critic.save('models/3vs1_three_check_max/model_critic_{}_{}.hdf5'.format(iters, iter_rewards[0]))
    env.reset()  # reset game after every iteration to reduce training wasted time.
    iters += 1
print("time taken to finish whole training: " + str(time.time() - start)) # prints at what time the code ends
env.close()
