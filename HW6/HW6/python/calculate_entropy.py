import math


def main():
    # p = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    # total = 36.0
    # entropy = 0
    # for i in range(len(p)):
    #     entropy += (p[i] / total) * (math.log2(p[i] / total))
    # print(entropy)
    # print(math.log2(36))
    # y = [0.375, 0.625]
    # total = 0
    # for i in range(len(y)):
    #     print(y[i] * (math.log2(y[i])))
    #     total += ((y[i]) * (math.log2(y[i])))
    # print(total)
    y = [0.15, 0.225, 0]
    Y = [y]
    y = [0.125, 0.3, 0.2]
    Y.append(y)
    print(Y)
    X = [0.275, 0.525, 0.2]
    Y_ = [0.375, 0.625]
    t_0 = Y[0][0]/ X[0]
    t_1 = Y[1][0]/ X[0]
    sum_0 = (t_0 * math.log2(t_0)) + (t_1 * math.log2(t_1))
    t_0 = Y[0][1] / X[1]
    t_1 = Y[1][1] / X[1]
    sum_1 = (t_0 * math.log2(t_0)) + (t_1 * math.log2(t_1))
    t_1 = Y[1][2] / X[2]
    sum_2 = (t_1 * math.log2(t_1))
    sum = [sum_0, sum_1, sum_2]
    t = [x*s for x, s in zip(X, sum)]
    print(t)
    total = t[0] + t[1]
    print(total)

if __name__ == "__main__":
    main()
