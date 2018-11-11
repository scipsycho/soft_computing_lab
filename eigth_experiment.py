from adaline import *

def transform_bipolar(val : list) -> list:
    ans = []
    for i in val:
        if i:
            ans.append(1)
        else:
            ans.append(-1)
    return ans



def generate():

    input_v = []
    output_v = []

    x = True
    y = True


    for i in range(4):

        t = x and (not y)

        input_v.append(transform_bipolar([x,y,True]))
        output_v.append(transform_bipolar([t]))

        if not y:
            x = not x
        y = not y


    return input_v, output_v

def train(opr: str, input_v:list, output_v:list, x: adaline_network, isTrain = True):

    error = 0
    for i in range(4):
        x.send(input_v[i])
        x.generate()
        error += x.lms_error(output_v[i])
        if isTrain:
            x.learn(output_v[i])

    x.show_network(opr+"Network")
    print(x.n_layers[0].weight_matrix[:,0])

    return error

#and not implementation using adaline
input_v, output_v = generate()

x = adaline_network(3,actn_fxn = identity)
epochs  = 6
for i in range(epochs):
    e = train('And_adaline',input_v,output_v,x)
    print("Error: {}".format(e))
e = train('And_adaline',input_v,output_v,x)
print("Error: {}".format(e))
