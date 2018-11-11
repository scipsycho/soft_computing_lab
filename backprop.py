from neural_network import *

class backprop_network(n_ff_network):

    def __init__(self, ff_layers, actn_fxn = sigmoid_binary, learning_rate = 0.2, initial_weights = 0.2):

        self.learning_rate = learning_rate
        self.actn_fxn = actn_fxn
        super(backprop_network,self).__init__(ff_layers,actn_fxn = actn_fxn, initial_weights = initial_weights)


    def lms_error(self, output_vector):
        pass

    def learn(self, output_vector):

        if self.output_vector is None:
            logging.critical("Output is not yet calculated. Cannot Continue!")
            return None

        output_vector = np.array(output_vector).flatten()

        if output_vector.shape != (self.ff_layers[self.num_layers-1],):
            logging.critical("Output is not of compatible size. Cannot Continue")
            return None


        input_layer = self.n_layers[0]

        delta = None
        new_delta = []
        for layer_i in reversed(self.n_layers):

            if delta is None:
                
