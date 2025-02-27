import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, sizes, optimizer="sgd"):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer

        if self.optimizer == "adam":
            self.m_weights = [np.zeros(w.shape) for w in self.weights]
            self.m_biases = [np.zeros(b.shape) for b in self.biases]
            self.v_weights = [np.zeros(w.shape) for w in self.weights]
            self.v_biases = [np.zeros(b.shape) for b in self.biases]


    def train(self, training_data,training_class, val_data, val_class, epochs, mini_batch_size, eta, lambda_reg, beta1, beta2, epsilon, decay_rate):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        iteration_index = 0
        eta_current = eta

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch " + str(j + 1))
            loss_avg = 0.0
            tp = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = self.backward_pass(output, mini_batch[1], Zs, As, lambda_reg)

                self.update_network(gw, gb, eta_current, beta1, beta2, epsilon, iteration_index)

                eta_current = eta * np.exp(-decay_rate * j)
                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output) + l2_reg(lambda_reg, self.weights, mini_batch_size)
                loss_avg += loss

                predictions = np.argmax(output, axis=0)
                true_labels = np.argmax(mini_batch[1], axis=0)
                tp += np.sum(predictions == true_labels)

            print("Epoch {} complete".format(j + 1))

            epoch_loss = loss_avg / len(mini_batches)
            epoch_accuracy = tp / n
            print("Train Loss: " + str(epoch_loss))
            print("Train Accuracy: " + str(epoch_accuracy))
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            val_loss, val_accuracy = self.eval_network(val_data, val_class, lambda_reg) 
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print()

        return train_losses, train_accuracies, val_losses, val_accuracies


    def eval_network(self, validation_data, validation_class, lambda_reg):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]

        outputs, _, _ = self.forward_pass(validation_data)
        val_loss = cross_entropy(validation_class, outputs) + l2_reg(lambda_reg, self.weights, n)

        predictions = np.argmax(outputs, axis=0)
        true_classes = np.argmax(validation_class, axis=0)
        tp = np.sum(predictions == true_classes)
        val_accuracy = tp / n
        
        print("Validation Loss: " + str(val_loss))
        print("Validation accuracy: "+ str(val_accuracy))

        return val_loss, val_accuracy

    def update_network(self, gw, gb, eta, beta1, beta2, epsilon, iteration_index):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            for i in range(len(self.weights)):
                self.m_weights[i] = (beta1 * self.m_weights[i]) + ((1 - beta1) * gw[i])
                self.m_biases[i] = (beta1 * self.m_biases[i]) + ((1 - beta1) * gb[i])

                self.v_weights[i] = (beta2 * self.v_weights[i]) + ((1 - beta2) * (gw[i] ** 2))
                self.v_biases[i] = (beta2 * self.v_biases[i]) + ((1 - beta2) * (gb[i] ** 2))

                m_w = self.m_weights[i] / (1 - (beta1 ** (iteration_index + 1)))
                m_b = self.m_biases[i] / (1 - (beta1 ** (iteration_index + 1)))
                v_w = self.v_weights[i] / (1 - (beta2 ** (iteration_index + 1)))
                v_b = self.v_biases[i] / (1 - (beta2 ** (iteration_index + 1)))

                self.weights[i] -= (eta * m_w) / ((np.sqrt(v_w) + epsilon))
                self.biases[i] -= (eta * m_b) / ((np.sqrt(v_b) + epsilon)) 
        else:
            raise ValueError('Unknown optimizer: '+ self.optimizer)


    def forward_pass(self, input):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        Zs = []
        As = [input]
        num_layers = len(self.weights)

        for i in range(num_layers):
            w = self.weights[i]
            b = self.biases[i]
        
            z = w.dot(As[-1]) + b
            Zs.append(z)

            if i == (num_layers - 1):
                a = softmax(z)
            else:
                a = sigmoid(z)
            As.append(a)
        
        return As[-1], Zs, As

    def backward_pass(self, output, target, Zs, activations, lambda_reg):
        num_layers = len(self.weights)
        gw = [np.zeros(w.shape) for w in self.weights]
        gb = [np.zeros(b.shape) for b in self.biases]
        N = output.shape[1]

        delta = softmax_dLdZ(output, target) * sigmoid_prime(Zs[-1])
        gw[-1] = delta.dot(activations[-2].T) + ((lambda_reg * self.weights[-1]) / N)
        gb[-1] = delta.sum(axis=1, keepdims=True)
        for i in range(2, num_layers + 1):
            delta = np.dot(self.weights[-i+1].T, delta) * sigmoid_prime(Zs[-i])
            gw[-i] = delta.dot(activations[-i-1].T) + ((lambda_reg * self.weights[-i]) / N)
            gb[-i] = delta.sum(axis=1, keepdims=True)

        return gw, gb
    

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

def sigmoid(z):
    z_clamped = np.maximum(z, -10)
    return 1.0 / (1.0 + np.exp(-z_clamped))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def l2_reg(lambda_reg, weights, N):
    return (lambda_reg * np.sum([np.sum(w**2) for w in weights])) / (2*N)

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()

    
def adjust_regularization_param(optimizer, learning_rate):
    models_reg = []
    lambda_regs = [0.0, 0.0003, 0.003, 0.03, 0.3, 1.0, 2.5, 5]
    for lambda_reg in lambda_regs:
        print(f"---MODEL WITH LAMBDA: {lambda_reg}---")
        net = Network([train_data.shape[0], 100, 10], optimizer=optimizer)
        _, train_accuracies, _, val_accuracies = net.train(
            train_data,
            train_class, 
            val_data, 
            val_class, 
            epochs=50, 
            mini_batch_size=64, 
            eta=learning_rate, 
            lambda_reg=lambda_reg,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            decay_rate=0
        )
        models_reg.append({
            'lambda_reg': lambda_reg,
            'val_accuracies': val_accuracies,
            'train_accuracies': train_accuracies
        })

    num_plots = len(models_reg)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 5*num_rows))
    for i, model in enumerate(models_reg):
        plt.subplot(num_rows, num_cols, i+1)
        plt.plot(range(1, len(model['train_accuracies']) + 1), model['train_accuracies'], label='Train Accuracy')
        plt.plot(range(1, len(model['val_accuracies']) + 1), model['val_accuracies'], label='Validation Accuracy')
        plt.title(f'Classification Accuracies Using Lambda: {model["lambda_reg"]}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    plt.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.tight_layout()
    plt.show()


def adjust_learning_rate(optimizer, lambda_reg):
    models_lr = []
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.5]
    for eta in learning_rates:
        print(f"---MODEL WITH LEARNING RATE: {eta}---")
        net = Network([train_data.shape[0], 100, 10], optimizer=optimizer)
        train_losses, _, val_losses, _ = net.train(
            train_data,
            train_class, 
            val_data, 
            val_class, 
            epochs=50, 
            mini_batch_size=64, 
            eta=eta, 
            lambda_reg=lambda_reg,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            decay_rate=0
        )
        models_lr.append({
            'learning_rate': eta,
            'val_losses': val_losses,
            'train_losses': train_losses
        })

    num_plots = len(models_lr)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 5*num_rows))
    for i, model in enumerate(models_lr):
        plt.subplot(num_rows, num_cols, i+1)
        plt.plot(range(1, len(model['train_losses']) + 1), model['train_losses'], label='Train Loss')
        plt.plot(range(1, len(model['val_losses']) + 1), model['val_losses'], label='Validation Loss')
        plt.title(f'Losses Using Learning Rate: {model["learning_rate"]}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    plt.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.tight_layout()
    plt.show()


def adjust_decay_rate(optimizer, learning_rate, lambda_reg):
    models_decay = []
    decay_rates = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99]
    for decay_rate in decay_rates:
        print(f"---MODEL WITH DECAY RATE: {decay_rate}---")
        net = Network([train_data.shape[0], 100, 10], optimizer=optimizer)
        train_losses, _, val_losses, _ = net.train(
            train_data,
            train_class, 
            val_data, 
            val_class, 
            epochs=50, 
            mini_batch_size=64, 
            eta=learning_rate, 
            lambda_reg=lambda_reg,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            decay_rate=decay_rate
        )
        models_decay.append({
            'decay_rate': decay_rate,
            'val_losses': val_losses,
            'train_losses': train_losses
        })

    num_plots = len(models_decay)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    plt.figure(figsize=(15, 5*num_rows))
    for i, model in enumerate(models_decay):
        plt.subplot(num_rows, num_cols, i+1)
        plt.plot(range(1, len(model['train_losses']) + 1), model['train_losses'], label='Train Loss')
        plt.plot(range(1, len(model['val_losses']) + 1), model['val_losses'], label='Validation Loss')
        plt.title(f'Losses Using Decay Rate: {model["decay_rate"]}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    plt.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.tight_layout()
    plt.show()


def adjust_optimizer():
    print(f"---SGD MODEL---")
    net_sgd = Network([train_data.shape[0], 100, 10], optimizer="sgd")
    train_losses_sgd, train_accuracies_sgd, val_losses_sgd, val_accuracies_sgd = net_sgd.train(
        train_data,
        train_class, 
        val_data, 
        val_class, 
        epochs=100, 
        mini_batch_size=64, 
        eta=0.01, 
        lambda_reg=0.03,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        decay_rate=0.1
    )
    
    print(f"---ADAM MODEL---")
    net_adam = Network([train_data.shape[0], 100, 10], optimizer="adam")
    train_losses_adam, train_accuracies_adam, val_losses_adam, val_accuracies_adam = net_adam.train(
        train_data,
        train_class, 
        val_data, 
        val_class, 
        epochs=100, 
        mini_batch_size=64, 
        eta=0.01, 
        lambda_reg=0.03,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        decay_rate=0.1
    )

    plt.figure(figsize=(12, 8))

    #Plot SGD losses
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses_sgd) + 1), train_losses_sgd, label='Train Loss - SGD', color='blue')
    plt.plot(range(1, len(val_losses_sgd) + 1), val_losses_sgd, label='Validation Loss - SGD', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('SGD Model - Losses')
    plt.grid(True)
    plt.legend()

    # Plot SGD accuracies
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracies_sgd) + 1), train_accuracies_sgd, label='Train Accuracy - SGD', color='blue')
    plt.plot(range(1, len(val_accuracies_sgd) + 1), val_accuracies_sgd, label='Validation Accuracy - SGD', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('SGD Model - Accuracies')
    plt.grid(True)
    plt.legend()

    # Plot Adam losses
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(train_losses_adam) + 1), train_losses_adam, label='Train Loss - Adam', color='green')
    plt.plot(range(1, len(val_losses_adam) + 1), val_losses_adam, label='Validation Loss - Adam', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Adam Model - Losses')
    plt.grid(True)
    plt.legend()

    # Plot Adam accuracies
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(train_accuracies_adam) + 1), train_accuracies_adam, label='Train Accuracy - Adam', color='green')
    plt.plot(range(1, len(val_accuracies_adam) + 1), val_accuracies_adam, label='Validation Accuracy - Adam', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Adam Model - Accuracies')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(train_data.shape[1] * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    
    np.random.seed(42)
    #adjust_regularization_param("sgd", learning_rate=0.001) # Best value: 0.03
    #adjust_learning_rate("sgd", lambda_reg=0.03) # Best value: 0.001 (without learning rate decay)
    #adjust_decay_rate("sgd", learning_rate=0.01, lambda_reg=0.03) # Best value: 0.1 (with learning rate 0.01)
    # Best values achieved using lambda_reg=0.03, learning_rate=0.01, decay_rate=0.1

    #adjust_regularization_param("adam", learning_rate=0.01) # Best value: 0.3
    #adjust_learning_rate("adam", lambda_reg=0.3) # Best value: 0.01 (without learning rate decay)
    #adjust_decay_rate("adam", learning_rate=0.01, lambda_reg=0.03) # Best value: 0.1 (with learning rate ...)
    # Best values achieved using lambda_reg=0.03, learning_rate=0.01, decay_rate=0.1

    adjust_optimizer()