from neural_network import *

class adaline_network(n_ff_network):

    def __init__(self, num_input_nodes, actn_fxn = sigmoid_binary, learning_rate = 0.2, initial_weights = 0.2):

        self.learning_rate = learning_rate
        super( adaline_network,self).__init__([num_input_nodes,1],actn_fxn = actn_fxn, initial_weights = initial_weights)


    def lms_error(self, output_vector):
        if self.output_vector is None:
            logging.critical("Output is not yet calculated. Cannot Continue!")
            return None

        output_vector = np.array(output_vector).flatten()

        if output_vector.shape != (self.ff_layers[1],):
            logging.warning("Output is not of compatible size. Cannot Continue")
            return None

        return (output_vector[0] - self.output_vector[0])**2
    def learn(self, output_vector):

        if self.output_vector is None:
            logging.critical("Output is not yet calculated. Cannot Continue!")
            return None

        output_vector = np.array(output_vector).flatten()

        if output_vector.shape != (self.ff_layers[1],):
            logging.critical("Output is not of compatible size. Cannot Continue")
            return None


        input_layer = self.n_layers[0]

        for i in range(len(output_vector)):
            output = output_vector[i]
            for j in range(input_layer.num_nodes):
                logging.debug("{} change in weights".format(self.learning_rate * (output - self.output_vector[i])*self.input_vector[j]))
                input_layer.weight_matrix[j][i] += self.learning_rate * (output - self.output_vector[i])*self.input_vector[j]
