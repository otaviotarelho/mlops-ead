import random

import numpy as np


# sizes = number of neurons
# sizes is a list, each element represents a layer
class Network(object):


    def __int__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Initialize biases and weights with random values using gaussian distribution
        # weights[0].shape = (3, 2)  # hidden layer ← input layer
        # weights[1].shape = (1, 3)  # output layer ← hidden layer

        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # creates a weight vector with size neuron x input entries
        # example : for 3 neurons in layer and 2 inputs, creates a 3x2 matrix
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    # training data is a tuple with input and expected output
    # eta is the learning rate
    # epochs is the number of iterations
    # mini_batch_size is the size of each mini batch to be used during the training
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta,
                                test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # iterate through each sample in the mini batch
        # x is the input, y is the expected output
        for x,y in mini_batch:
            # compute the gradient for the sample
            # delta_nabla_b and delta_nabla_w are the gradients for the sample
            # nabla_b and nabla_w are the accumulated gradients for the mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases using gradient descent formula
        # eta is the learning rate
        # divide by len(mini_batch) to get the sample size for the mini batch
        # update rule: w = w - (eta/m) * nabla_w
        # where m is the size of the mini batch
        # same for biases
        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    # x is the input, y is the expected output
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        # it's in a zip because we need to iterate through both biases and weights for each layer
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # compute the error for the output layer
        # delta = the error
        # nabla_b[-1] = delta for the last layer
        # nabla_w[-1] = delta * activation of previous layer (transposed)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # iterate through the layers backwards for the hidden layers update
        # l = 2 means the second last layer
        # -l means the l-th layer from the end
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # return the gradient for the biases and weights
        return nabla_b, nabla_w

    def cost_derivative(self, param, y):
        pass

    def evaluate(self, test_data):
        pass


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
