# # This is the BACKWARD PROPAGATION function
# def backward_prop(model,cache,y):

#     # Load parameters from model
#     W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
    
#     # Load forward propagation results
#     a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
    
#     # Get number of samples
#     m = y.shape[0]
    
#     # Calculate loss derivative with respect to output
#     dz3 = loss_derivative(y=y,y_hat=a3)

#     # Calculate loss derivative with respect to second layer weights
#     dW3 = 1/m*(a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
    
#     # Calculate loss derivative with respect to second layer bias
#     db3 = 1/m*np.sum(dz3, axis=0)
    
#     # Calculate loss derivative with respect to first layer
#     dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2))
    
#     # Calculate loss derivative with respect to first layer weights
#     dW2 = 1/m*np.dot(a1.T, dz2)
    
#     # Calculate loss derivative with respect to first layer bias
#     db2 = 1/m*np.sum(dz2, axis=0)
    
#     dz1 = np.multiply(dz2.dot(W2.T),tanh_derivative(a1))
    
#     dW1 = 1/m*np.dot(a0.T,dz1)
    
#     db1 = 1/m*np.sum(dz1,axis=0)
    
#     # Store gradients
#     grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
#     return grads
