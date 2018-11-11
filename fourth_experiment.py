from perceptron import *


def transform_bipolar(val : list) -> list:
    ans = []
    for i in val:
        if i:
            ans.append(1)
        else:
            ans.append(-1)
    return ans



def generate(func):

    input_v = []
    output_v = []

    x = True
    y = True

    func = func.lower()

    for i in range(4):

        t = x and y

        input_v.append(transform_bipolar([x,y,True]))
        output_v.append(transform_bipolar([t]))

        if not y:
            x = not x
        y = not y


    return input_v, output_v

def train(opr: str):

    input_v, output_v = generate(opr)
    x = perceptron_network(3)

    for i in range(4):
        x.send(input_v[i])
        x.generate()
        x.learn(output_v[i])

    x.show_network(opr+"Network")
    print(x.n_layers[0].weight_matrix[:,0])

#and implementation

train('And_perceptron')
