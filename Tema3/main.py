import pickle, gzip, numpy as np
import random
from matplotlib import pyplot as plt

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_size = input_layer_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_layer_size
        self.weights = []
        self.weights.append(np.random.randn(self.hidden_size, self.input_size) / np.sqrt(self.input_size))
        self.weights.append(np.random.randn(self.output_size, self.hidden_size) / np.sqrt(self.hidden_size))
        self.biases = []
        self.biases.append(np.random.rand(self.hidden_size, 1))
        self.biases.append(np.random.rand(self.output_size, 1))

    def train(self, train_set, valid_set, test_set, iterations_number, batch_size, learning_rate,momentum_coefficient,l2_constant):
        data = train_set[0]
        labels = train_set[1]
        data_len = len(data)
        batches_number = data_len // batch_size
        l2_term = 2 * l2_constant/data_len
        for iteration in range(iterations_number):
            permutation = [i for i in range(data_len)]
            random.shuffle(permutation)
            for i in range(batches_number):
                batch_weights_update = [np.zeros(w.shape) for w in self.weights]
                batch_biases_update = [np.zeros(b.shape) for b in self.biases]
                for j in range(batch_size):
                    index=permutation[i * batch_size + j]
                    element = data[index].reshape(len(data[index]),1)
                    annotation = labels[index]
                    layers_sums, layers_activations = self.feedforward(element)
                    weights_adjustments, biases_adjustments = self.backpropagation(element, annotation, layers_sums, layers_activations)
                    for k in range(len(weights_adjustments)):
                        batch_weights_update[k] += weights_adjustments[k]
                        batch_biases_update[k] += biases_adjustments[k]
                friction_weights = [np.ones(w.shape) for w in self.weights]
                friction_biases = [np.ones(b.shape) for b in self.biases]
                for k in range(len(batch_weights_update)):
                    friction_weights[k] = momentum_coefficient * friction_weights[k] - learning_rate * batch_weights_update[k]
                    friction_biases = momentum_coefficient * friction_biases[k] - learning_rate * batch_biases_update[k]
                    self.weights[k] = self.weights[k] + friction_weights[k]-learning_rate*l2_term*self.weights[k]
                    self.biases[k] = self.biases[k] + friction_biases[k]-learning_rate*l2_term*self.biases[k]
            print(f"Finished epoch: {iteration+1}")
            self.test(test_set)

    def feedforward(self, element):
        previous_layer_output=element
        layers_activations = [previous_layer_output]
        layers_sums = []
        for i in range(len(self.weights)):
            current_layer_sum=np.dot(self.weights[i],previous_layer_output)+self.biases[i]
            layers_sums.append(current_layer_sum)
            if i+1 != len(self.weights):
                previous_layer_output=sigmoid(current_layer_sum)
            else:
                previous_layer_output=softmax(current_layer_sum)
            layers_activations.append(previous_layer_output)
        return [layers_sums, layers_activations]

    def backpropagation(self, element, annotation, layers_sums, layers_activations):
        weights_adjustments = [np.zeros(w.shape) for w in self.weights]
        biases_adjustments = [np.zeros(b.shape) for b in self.biases]
        delta = layers_activations[-1] - identity_matrix[annotation].reshape(len(identity_matrix), 1)
        weights_adjustments[-1] = np.dot(delta, layers_activations[-2].transpose())
        biases_adjustments[-1] = delta
        for i in range(2, len(self.weights) + 1):
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_derivative(layers_sums[-i]) 
            weights_adjustments[-i] = np.dot(delta, layers_activations[-i - 1].transpose()) 
            biases_adjustments[-i] = delta 
        return [weights_adjustments, biases_adjustments]

    def feedforward_one(self, element):
        h_layer_sum = np.dot(self.weights[0], element) + self.biases[0]
        h_layer_output = sigmoid(h_layer_sum)
        o_layer_sum = np.dot(self.weights[1], h_layer_output) + self.biases[1]
        o_layer_output = softmax(o_layer_sum)
        return o_layer_output

    def test(self, test_set):
        wrong_classified_counter = 0
        data = test_set[0]
        labels = test_set[1]
        for i in range(len(data)):
            element = data[i].reshape(len(data[i]),1)
            output_layer = self.feedforward_one(element)
            biggest = np.argmax(output_layer)
            if identity_matrix[labels[i]][biggest]!=1:
                wrong_classified_counter += 1
        print(f"Accuracy : {100*(len(data)-wrong_classified_counter)/len(data)}%")



if __name__ == '__main__' :
    identity_matrix = np.array([1 if i == j else 0 for i in range(10) for j in range(10)]).reshape(10,10)
    epochs = 30
    learning_rate = 0.02
    batch_size = 10
    momentum_coefficient=0.000005
    l2_constant=0.2
    rn = NeuralNetwork(len(train_set[0][0]), 100, 10)
    rn.train(train_set, valid_set, test_set, epochs, batch_size, learning_rate,momentum_coefficient,l2_constant)
