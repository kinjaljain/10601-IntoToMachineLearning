import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

LEARNING_RATE = 0.1

def main():
    if len(sys.argv) < 9:
        print("Please give formatted_train_out, formatted_validation_out, formatted_test_out, dict_input, "
              "train_out_labels, test_out_labels, metrics_out, and epochs respectively in commandline arguments.")

    start = time.time()
    formatted_train_out = sys.argv[1]
    formatted_validation_out = sys.argv[2]
    formatted_test_out = sys.argv[3]
    dict_input = sys.argv[4]
    train_out_labels = sys.argv[5]
    test_out_labels = sys.argv[6]
    metrics_out = sys.argv[7]
    epochs = int(sys.argv[8])

    with open(dict_input, 'r') as f:
        dict_input = f.readlines()
        word_dict = {}
        for line in dict_input:
            key, value = line.split()
            word_dict[key.strip()] = int(value)
    with open(formatted_train_out, 'r') as f:
        train_data = f.readlines()
    with open(formatted_validation_out, 'r') as f:
        validation_data = f.readlines()
    with open(formatted_test_out, 'r') as f:
        test_data = f.readlines()

    theta = np.zeros(len(word_dict)+1)
    X_train, Y_train = get_data(train_data)
    X_validation, Y_validation = get_data(validation_data)
    X_test, Y_test = get_data(test_data)
    # theta = train_logistic_regression(X_train, Y_train, theta, epochs)
    theta, train_loss, validation_loss = train_logistic_regression_valid(X_train, Y_train, X_validation, Y_validation, theta, epochs)
    Y_train_predicted = predict(X_train, theta)
    Y_test_predicted = predict(X_test, theta)
    train_error = calculate_error(Y_train_predicted, Y_train)
    test_error = calculate_error(Y_test_predicted, Y_test)

    with open(train_out_labels, 'w') as f:
        for i in range(len(Y_train_predicted)):
            f.write(str(Y_train_predicted[i]))
            f.write('\n')
    with open(test_out_labels, 'w') as f:
        for i in range(len(Y_test_predicted)):
            f.write(str(Y_test_predicted[i]))
            f.write('\n')
    with open(metrics_out, 'w') as f:
        f.write("error(train): {}\n".format(str.format('{0:.6f}', train_error)))
        f.write("error(test): {}\n".format(str.format('{0:.6f}', test_error)))
    end = time.time()
    print(end-start)

    print(train_loss)
    print(validation_loss)
    plt.plot(range(len(train_loss[:200])), train_loss[:200], label="Train Loss")
    plt.plot(range(len(validation_loss[:200])), validation_loss[:200], label="Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Negative Log Likelihood")
    plt.legend()
    plt.show()

def calculate_error(Y_predicted, Y):
    count = 0.0
    for i in range(len(Y)):
        if Y_predicted[i] != Y[i]:
            count += 1
    return count/float(len(Y))

def train_logistic_regression(X_train, Y_train, theta, epochs):
    for i in range(epochs):
        for j in range(len(X_train)):
            theta = update_theta_step_sgd(theta, X_train, Y_train, j)
        # train_loss = calculate_loss_function(theta, X_train, Y_train)
        # print("Train loss after epoch {}: {}\n".format(i, train_loss))
    return theta

def train_logistic_regression_valid(X_train, Y_train, X_validation, Y_validation, theta, epochs):
    train_loss, validation_loss = [], []
    for i in range(epochs):
        for j in range(len(X_train)):
            theta = update_theta_step_sgd(theta, X_train, Y_train, j)
        train_loss.append(calculate_loss_function(theta, X_train, Y_train))
        validation_loss.append(calculate_loss_function(theta, X_validation, Y_validation))
    return theta, train_loss, validation_loss

def predict(X, theta):
    Y = []
    for i in range(len(X)):
        theta_x = calculate_dot_product(theta, X[i])
        exponent = math.exp(theta_x)
        if (exponent/(1.0 + exponent)) > 0.5:
            Y.append(1)
        else:
            Y.append(0)
    return Y

def get_data(data_lines):
    X, Y = [], []
    for data_line in data_lines:
        line_data = data_line.split('\t')
        y = int(line_data[0])
        Y.append(y)
        x = {0: 1}
        parameters = line_data[1:]
        for parameter in parameters:
            vals = parameter.split(':')
            x[int(vals[0])+1] = int(vals[1])
        X.append(x)
    return X, Y

def update_theta_step_sgd(theta, X, Y, i):
    theta_x = calculate_dot_product(theta, X[i])
    gradient = np.zeros(len(theta))
    for key in X[i].keys():
        term1 = -1 * (X[i])[key]
        exponent = math.exp(theta_x)
        term2 = Y[i] - (exponent/(1.0 + exponent))
        gradient[key] = term1 * term2
    theta = theta - LEARNING_RATE * gradient
    return theta

def calculate_loss_function(theta, X, Y):
    sum = 0.0
    for i in range(len(X)):
        theta_x = calculate_dot_product(theta, X[i])
        term1 = -1 * Y[i] * theta_x
        term2 = math.log(1.0 + math.exp(theta_x))
        sum += (term1 + term2)
    return sum/len(X)

def calculate_dot_product(theta, x):
    sum = 0.0
    for key in x.keys():
        sum += theta[key] * x[key]
    return sum

if __name__ == "__main__":
    main()
