from random import *
from math import *

class Neuron():
    def __init__(self, size, alpha):
        self.alpha = alpha
        self.weights = [random() for _ in range(size)]

    def feed_forward(self, inputs):
        return self.activate(sum([x*y for x, y in zip(inputs, self.weights)]))

    def activate(self, x):
        return 1 / (1 + (e**(-x)))

    def train(self, inputs, desired_output):
        output = self.feed_forward(inputs)
        error = (desired_output - output)
        for x in range(len(self.weights)):
            self.weights[x] += error * self.alpha * inputs[x]

def check(x, y):
    return int(3 * x < y)

n = Neuron(2, 0.01)

for x in [[random() * 21 - 10, uniform(-10, 10)] for _ in range(1000000)]:
    n.train(x, check(x[0], x[1]))
print(n.feed_forward([1, 3])) # Should be about 0.5
print(n.feed_forward([1, 4])) # Should be about 1
print(n.feed_forward([1, 2])) # Should be about 0
