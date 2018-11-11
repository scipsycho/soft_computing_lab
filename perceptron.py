from neural_network import *

logging.basicConfig(filename='./log.txt',level=logging.DEBUG)

class perceptron_network(n_ff_network):

    def __init__(self, num_input_nodes, num_output_nodes = 1, theta = 0, learning_rate = 1 , initial_weight = None):
        self.learning_rate = learning_rate
        self.theta = theta
        super(perceptron_network,self).__init__([num_input_nodes, num_output_nodes],initial_weight,actn_fxn = self.step)

    def step(self,x) -> int:
        logging.debug("Step function({}) called".format(x))
        if x > self.theta:
            return 1
        elif x < self.theta:
            return -1
        else:
            return 0

    def learn(self,output_vector):

        if self.output_vector is None:
            logging.critical("Output is not yet calculated. Cannot Continue!")
            return None

        output_vector = np.array(output_vector).flatten()

        if output_vector.shape != (self.ff_layers[1],):
            logging.critical("Output is not of compatible size. Cannot Continue")
            return None

        #comparing computed output with actual output
        if np.equal(output_vector,self.output_vector):
            return
            
        input_layer = self.n_layers[0]

        for i in range(len(output_vector)):
            output = output_vector[i]
            for j in range(input_layer.num_nodes):
                input_layer.weight_matrix[j][i] += self.learning_rate*output*self.input_vector[j]
