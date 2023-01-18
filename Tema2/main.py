import numpy as np
import pickle, gzip, numpy
with gzip.open('mnist.pkl.gz','rb') as fd : 
    train_set, valid_set, test_set = pickle.load(fd, encoding = 'latin')



class Perceptron:
    def __init__(self, data_set, weights: list[int] = None, aim: int = 0, epoch: int = 30, learning_rate = 0.1):
        self.images = np.array(data_set[0])
        self.tags = data_set[1]
        self.bias = 0
        self.epoch = epoch
        self.aim = aim
        if weights is None:
            self.weights = np.zeros(self.images.shape[1])
        else:
            self.weights = weights
        self.learning_rate = learning_rate

    def learn(self):
        for epoch in range(self.epoch):
            print(f"Epoch: {epoch}\n")
            for perceptronNumber in range(self.images.shape[0]):
                image = np.array(self.images[perceptronNumber])
                label = self.tags[perceptronNumber] == self.aim
                z = np.dot(image, self.weights) + self.bias
                output = self.activation_function(z)
                self.weights += (label - output) * image * self.learning_rate
                self.bias += (label-output) * self.learning_rate
        print("\nFinished learning :D \n")

    def guessNr(self, image):
        out = np.dot(image, self.weights) + self.bias
        return self.activation_function(out)

    def accuracy(self, model_set):
        aim_target = np.array([1 if x == 0 else 0 for x in model_set[1]])
        predictions = self.guessNr(model_set[0])
        correct = 0
        for perceptronNumber in range(aim_target.shape[0]):
            if aim_target[perceptronNumber] == predictions[perceptronNumber]:
                correct += 1
        return correct/aim_target.shape[0]

    def guessMultipleNr(self, image):
        out = np.dot(image, self.weights) + self.bias
        return (out, self.activation_function(out))


    def activation_function(self, z):
        return np.where(z >= 0, 1, 0)

def verify(all_perceptrons, image, label):
    guess = -1
    max_chance = -1
    chance_for_correct_label = 0
    for perceptronNumber, perceptron in enumerate(all_perceptrons):
        guessed, is_ok = perceptron.guessMultipleNr(image)
        if is_ok == 1 and guessed > max_chance:
            max_chance = guessed
            guess = perceptronNumber
        if perceptronNumber == label:
            chance_for_correct_label = guessed
    if label == guess:
        return True
    if guess == -1:
        return False
    print(f"Didn't guess correctly for digit {label}.")
    return False

def main() : 
    all_perceptrons = []
    for perceptronNumber in range(10):
        print(f"Training perceptron for digit {perceptronNumber}")
        perceptronn = Perceptron(train_set, epoch=50, aim=perceptronNumber, learning_rate=0.05)
        perceptronn.learn()
        print(f"Perceptron for digit : {perceptronn.aim} accuracy in valid_set : {perceptronn.accuracy(valid_set)}")
        print(f"Perceptron for digit : {perceptronn.aim} accuracy in test_set : {perceptronn.accuracy(test_set)}")
        all_perceptrons.append(perceptronn)
    images, labels = test_set
    accuracy = 0
    for perceptronNumber in range(labels.shape[0]):
        image, label = (images[perceptronNumber], labels[perceptronNumber])
        accuracy += verify(all_perceptrons, image, label)
    print(f"Total sample data in test set: {labels.shape[0]} \nAmount of numbers predicted correctly : {accuracy}")
    print(f"Algorithm accuracy is: {(accuracy/labels.shape[0])*100}%\n")

main()