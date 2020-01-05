import sys
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, loss_function, num_input_units, num_output_units, learning_rate):
        self.layers = []
        self.loss_function = loss_function
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, layer_input):
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
        return layer_input

    def calculate_error(self, y_actual, y_predicted):
        if self.loss_function == "nll":
            nll = NegativeLogLikelihood(y_actual, y_predicted)
            return nll.calculate()

    def calculate_gradient(self, y_actual, y_predicted):
        if self.loss_function == "nll":
            nll = NegativeLogLikelihood(y_actual, y_predicted)
            return nll.gradient()

    def backward(self, layer_gradient):
        for layer in reversed(self.layers):
            layer_gradient = layer.backward(layer_gradient)

    def apply(self):
        for layer in self.layers:
            layer.apply()

    def train(self, x_train, y_train, x_test, y_test, num_epochs):
        avg_train_loss = []
        avg_test_loss = []
        y_train_ = []
        y_test_ = []
        for e in range(num_epochs):
            train_loss = 0
            test_loss = 0
            for i in range(len(x_train)):
                y_predicted = self.forward(x_train[i])
                gradient = self.calculate_gradient(y_train[i], y_predicted)
                self.backward(gradient)
                self.apply()
            for i in range(len(x_train)):
                y_train_predicted = self.forward(x_train[i])
                train_loss += self.calculate_error(y_train[i], y_train_predicted)
                if e == (num_epochs - 1):
                    y_train_.append(np.argmax(y_train_predicted).tolist())
            for i in range(len(x_test)):
                y_test_predicted = self.forward(x_test[i])
                test_loss += self.calculate_error(y_test[i], y_test_predicted)
                if e == (num_epochs - 1):
                    y_test_.append(np.argmax(y_test_predicted).tolist())
            avg_train_loss.append(train_loss.max() / len(x_train))
            avg_test_loss.append(test_loss.max() / len(x_test))
        return avg_train_loss, avg_test_loss, y_train_, y_test_


class LinearLayer:
    def __init__(self, input_size, hidden_size, weight_flag, bias, learning_rate):
        self.weight = initialise_np_array(hidden_size, input_size, weight_flag)
        self.bias = bias
        self.learning_rate = learning_rate
        self.grad_weight = []
        self.grad_bias = []
        self.input = []

    def forward(self, x):
        self.input.append(x)
        return self.bias + np.matmul(self.weight, x)

    def backward(self, grad_a):
        x = self.input.pop()
        self.grad_weight.append(np.matmul(grad_a, x.T))
        self.grad_bias.append(grad_a)
        return np.matmul(self.weight.T, grad_a)

    def apply(self):
        self.grad_weight = (np.sum(self.grad_weight, axis=0)) / len(self.grad_weight)
        self.grad_bias = (np.sum(self.grad_bias, axis=0)) / len(self.grad_bias)
        self.weight = self.weight - (self.learning_rate * self.grad_weight)
        self.bias = self.bias - (self.learning_rate * self.grad_bias)
        self.grad_weight = []
        self.grad_bias = []


class ActivationLayer:
    def __init__(self, function_flag):
        self.function = function_flag
        self.output = []

    def forward(self, a):
        # can add more functions in future
        assert (self.function == 1)
        if self.function == 1:
            # sigmoid
            output = 1 / (1 + np.exp(-a))
            self.output.append(output)
            return output

    def backward(self, grad_z):
        assert (self.function == 1)
        if self.function == 1:
            # sigmoid
            z = self.output.pop()
            return grad_z * z * (1 - z)

    def apply(self):
        pass


class OutputLayer:
    def __init__(self, function_flag):
        self.function = function_flag
        # self.output = []

    def forward(self, b):
        # can add more functions in future
        assert (self.function == 1)
        if self.function == 1:
            # softmax output
            output = np.exp(b) / np.sum(np.exp(b))
            # self.output.append(output)
            return output

    def backward(self, loss_grad):
        assert (self.function == 1)
        if self.function == 1:
            # softmax output
            return loss_grad

    def apply(self):
        pass


class NegativeLogLikelihood:
    def __init__(self, y, y_):
        self.y = y
        self.y_ = y_

    def calculate(self):
        return -np.matmul(self.y.T, np.log(self.y_))

    def gradient(self):
        return self.y_ - self.y


def initialise_np_array(num_rows, num_cols, flag):
    assert (flag == 1 or flag == 2)
    if flag == 1:
        # initialize with uniform distribution between [-0.1, 0.1]
        return np.random.uniform(low=-0.1, high=0.1, size=(num_rows, num_cols))
    elif flag == 2:
        # initialize with zeros
        return np.zeros(shape=(num_rows, num_cols), dtype=np.float64, order='C')


def fetch_x_y(filename, num_classes):
    with open(filename, 'r') as f:
        train_data = f.readlines()
    x = []
    y = []
    for line in train_data:
        data = line.rstrip().split(',')
        y_i = np.eye(num_classes, dtype=int)[int(data[0])]
        y.append(np.reshape(y_i, (y_i.shape[0], 1)))
        x_i = np.array(data[1:], dtype=int)
        x.append(np.reshape(x_i, (x_i.shape[0], 1)))
    # x_y = np.array(data)
    # x = x_y[:, 1:]
    # y = x_y[:, 0]
    # x = np.reshape(x_y[:, 1:], (x_y[:, 1:].shape[0], 1))
    # y = np.reshape(x_y[:, 0], (x_y[:, 0].shape[0], 1))
    return x, y


def error(y_actual, y_predicted):
    model_error = 0
    for i in range(len(y_actual)):
        if y_actual[i][y_predicted[i]] != 1:
            model_error += 1
    return model_error / float(len(y_actual))


def main():
    if len(sys.argv) < 10:
        print("Please give train_input file, test_input file, train_out file, test_out file, "
              "metrics_out file, num_epochs, num_hidden_units, init_flag, and learning_rate "
              "respectively in commandline arguments")
    train_input_file = sys.argv[1]
    test_input_file = sys.argv[2]
    train_output_file = sys.argv[3]
    test_output_file = sys.argv[4]
    metrics_output_file = sys.argv[5]
    num_epochs = int(sys.argv[6])
    num_hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    # num_output_classes = 10
    train_x, train_y = fetch_x_y(train_input_file, 10)
    test_x, test_y = fetch_x_y(test_input_file, 10)

    num_input_units = len(train_x[0])
    num_output_units = len(train_y[0])
    assert num_output_units == 10

    nn = NeuralNetwork(loss_function="nll", num_input_units=num_input_units,
                       num_output_units=num_output_units, learning_rate=learning_rate)
    layer_1 = LinearLayer(num_input_units, num_hidden_units, init_flag, 0, learning_rate)
    hidden_layer_1 = ActivationLayer(1)
    layer_2 = LinearLayer(num_hidden_units, num_output_units, init_flag, 0, learning_rate)
    output_layer = OutputLayer(1)
    nn.add_layer(layer_1)
    nn.add_layer(hidden_layer_1)
    nn.add_layer(layer_2)
    nn.add_layer(output_layer)
    avg_train_loss, avg_test_loss, y_train_, y_test_ = nn.train(train_x, train_y, test_x, test_y, num_epochs)
    train_error = error(train_y, y_train_)
    test_error = error(test_y, y_test_)

    x = range(1, 101)
    plt.plot(x,
             [t.item() for t in avg_train_loss], label='train')
    plt.plot(x,
             [t.item() for t in avg_test_loss], label='test')

    plt.xlabel('Number of epochs')
    plt.ylabel('Average cross-entropy loss')
    plt.title("Average cross-entropy vs number of epochs : learning rate-%s" % learning_rate)
    plt.axis([0, 2.25, 1, 100])
    plt.legend()
    plt.show()

    # avg_train_losses = [0.5474677053289023, 0.13620188311702486, 0.0558556963362208, 0.0465599072131755, 0.045575001415795985]
    # avg_test_losses = [0.7427750671608365, 0.5915195983034586, 0.44266105500791647, 0.43176250665349153, 0.4329911233108805]
    #
    # x = [5, 20, 50, 100, 200]
    # plt.plot(x, avg_train_losses, marker='o', label='train')
    # plt.plot(x, avg_test_losses, marker='o', label='test')
    #
    # plt.xlabel('Number of hidden units')
    # plt.ylabel('Average cross-entropy')
    # plt.title("Average cross-entropy vs number of hidden units")
    # plt.legend()
    # plt.show()

    with open(train_output_file, 'w') as f:
        for i in range(len(y_train_)):
            f.write(str(y_train_[i]) + "\n")

    with open(test_output_file, 'w') as f:
        for i in range(len(y_test_)):
            f.write(str(y_test_[i]) + "\n")

    with open(metrics_output_file, 'w') as f:
        for i in range(len(avg_train_loss)):
            f.write("epoch=%s crossentropy(train): %s\n" % (i + 1, avg_train_loss[i].item()))
            f.write("epoch=%s crossentropy(test): %s\n" % (i + 1, avg_test_loss[i].item()))
            # f.write("epoch=%s crossentropy(train): %s\n" % (i + 1, np.asscalar(avg_train_loss[i])))
            # f.write("epoch=%s crossentropy(test): %s\n" % (i + 1, np.asscalar(avg_test_loss[i])))
        f.write("error(train): %s\n" % train_error)
        f.write("error(test): %s\n" % test_error)


if __name__ == "__main__":
    main()
