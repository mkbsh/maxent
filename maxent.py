# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import math
import random
from scipy import misc
from collections import Counter

class MaxEnt(Classifier):

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient"""
        # Some important parameters.

        instances = train_instances + dev_instances

        words = [feature for inst in instances for feature in inst.features()]
        self.features = set([word for word, word_count in Counter(words).most_common(1000)])
        self.features.add('BIAS')
        self.labels = set([inst.label for inst in instances])

        # Parameters should be initialized as 0.
        self.weights = dict()

        for feature in self.features:
            self.weights[feature] = {}
            for label in self.labels:
                self.weights[feature][label] = 0

        # Cache the features into the documents.
        for instance in instances:
            self.vectorize(instance)

        converged = False
        max_accuracy = 0
        no_change = 0
        best_weights = {}

        while not converged:
            # Chop up training set into batches of size batch_size.
            random.shuffle(train_instances)
            batches = [train_instances[i:i + batch_size] for i in range(0, len(train_instances), batch_size)]
            for batch in batches:
                gradient = self.compute_gradient(batch)
                self.gradient_descent(gradient, learning_rate)
            accuracy = self.accuracy(dev_instances)
            log_likelihood = self.log_likelihood(dev_instances)
            print("Accuracy: ", accuracy, "| Negative log-likelihood: ", log_likelihood)
            no_change += 1
            if accuracy > max_accuracy:
                best_weights = self.weights
                max_accuracy = accuracy
                no_change = 0
            if no_change > 3:
                converged = True
        self.weights = best_weights

    # Compute the gradient for each (feature, label).
    def compute_gradient(self, batch):
        gradient = dict()
        # Initialize.
        for feature in self.features:
            gradient[feature] = {}
            for label in self.labels:
                gradient[feature][label] = 0

        for instance in batch:
            posterior = self.compute_posterior(instance)
            for feature in instance.feature_vector:
                gradient[feature][instance.label] += 1
                for label in self.labels:
                    gradient[feature][label] -= posterior[label]

        return gradient

    # Compute the posterior probability of P(label | feature) for each (feature, label).
    def compute_posterior(self, instance):
        posterior = {}
        denominator = []
        for label in self.labels:
            posterior[label] = sum(self.weights[feature][label] for feature in instance.feature_vector)
            denominator.append(posterior[label])
        for label in self.labels:
            posterior[label] = math.exp(posterior[label] - misc.logsumexp(denominator))

        return posterior

    # Update parameters using gradient descent.
    def gradient_descent(self, gradient, learning_rate):
        for feature in self.features:
            for label in self.labels:
                self.weights[feature][label] += gradient[feature][label] * learning_rate

    def classify(self, instance):
        if (instance.feature_vector == []):
            self.vectorize(instance)

        posterior = self.compute_posterior(instance)

        if posterior != {}:
            return max(posterior, key=lambda x: posterior[x])
        else:
            return random.choice(self.labels)

    # For debugging.
    def print_matrix(self, matrix, string):
        print(string, " MATRIX")
        print(matrix)

    def vectorize(self, instance):
        for feature in instance.features():
            if feature in self.features:
                instance.feature_vector.append(feature)
        instance.feature_vector.append('BIAS')

    def accuracy(self, dev):
        correct = [self.classify(x) == x.label for x in dev]
        return float(sum(correct)) / len(correct)

    def log_likelihood(self, dev):
        sum = 0
        for instance in dev:
            posterior = self.compute_posterior(instance)
            if posterior[instance.label] != 0:
                sum += math.log(posterior[instance.label])
        return sum
