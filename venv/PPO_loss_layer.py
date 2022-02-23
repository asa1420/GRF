from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class PPO_loss_layer(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dim = output_dims
        super(PPO_loss_layer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape= (input_shape[1], self.output_dim), initializer= 'normal', trainable= True)
        super(PPO_loss_layer, self).build(input_shape) # Be sure to call this at the end
    def call(self, inputs, *args, **kwargs):
        return K.dot(inputs)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
