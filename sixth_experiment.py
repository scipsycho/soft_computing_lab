from neural_network import *

class mcpitts(n_ff_network):

    def __init__(self, num_input_nodes, initial_weights, theta = 0):
        self.theta = theta
        initial_weights = np.array(initial_weights).flatten()

        if initial_weights.shape[0] != num_input_nodes:
            logging.critical("Size of Intial Weight array incompatible with number of nodes. Cannot Continue")
            exit()
        initial_weights = initial_weights.reshape((1,num_input_nodes,1))
        super(mcpitts,self).__init__([num_input_nodes,1],actn_fxn = self.stair, initial_weights = initial_weights)

    def stair(x):
        if x>=self.theta:
            return 1
        else:
            return 0


mcpitts_network = mcpitts(8, initial_weights = [ 1, 1, 1, 1, 1, -1, -1, -1])

mcpitts_network.show_network("MC Pitts Neural Network")
