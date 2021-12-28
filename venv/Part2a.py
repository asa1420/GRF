import gfootball.env as football_env
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def get_model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    n_actions = output_dims

    # Use MobileNet feature extractor to process input image
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense( activation='softmax', name='predictions')(x)

    # Define model
    model = Model(inputs=[state_input], outputs=[out_actions]) # the model takes as input the current state and outputs a list of actions
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
for itr in range(ppo_steps): # collect 128 interactions with the game
    observation, reward, done, info = env.step(env.action_space.sample()) # this takes a random action in the game and gives as output the observation, reward, and other info, as well as if the game is done.
    if done:
        env.reset() # reset if the game is done
env.close()