import numpy as np

# load tiny images' net
def loadTiny(net):
    layers = [] # set layers
    for k, v in net.params.items(): # load layers
        layers.append(k)
    params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in layers} # load parameters
    return (layers, params)

# load large images's net
def loadLarge(net):
    layers = [] # set layers
    for k, v in net.params.items(): # load layers
        layers.append(k)
    dim1_w = [0 for i in layers] # initialise the dimension of weights
    dim2_w = [0 for i in layers]
    dim3_w = [0 for i in layers]
    dim4_w = [0 for i in layers]
    dim_b = [0 for i in layers] # initialise the dimension of bias
    for pr, i in zip(layers, range(len(layers))): # save the dimension of weights and bias
        dim1_w[i], dim2_w[i], dim3_w[i], dim4_w[i] = net.params[pr][0].data.shape
        dim_b[i] = net.params[pr][1].data.shape[3]
    params = {pr: (np.zeros((d1,d2,d3,d4)), np.zeros((1,1,1,db))) for pr, d1, d2, d3, d4, db in zip(layers, dim1_w, dim2_w, dim3_w, dim4_w, dim_b)} # initialize weights and bias
    return (layers, params)
            
# trans function is used to transform weights in fc6 layer
def trans(params, N, H, W):
    params_new = np.zeros((N,H,W))
    for i in range(N):
        temp = np.insert(np.insert(params[i,:,:], slice(1,None,1), 0, axis=1), slice(1,None,1), 0, axis=0) # extend weights
        params_new[i,:,:] = temp
    return params_new

def rotateArray(data):
    N, H, W = data.shape
    data_new = np.zeros((N,H,W))
    for i in range(N):
        #data_new[i,:,:] = np.flipud(data[i,:,:]) # Flip an array vertically
        #data_new[i,:,:] = np.fliplr(data[i,:,:]) # Flip an array horizontally
        #data_new[i,:,:] = np.fliplr(np.flipud(data[i,:,:])) # Flip an array centrally
        data_new[i,:,:] = np.transpose(data[i,:,:]) # Flip an array diagonally
    return data_new