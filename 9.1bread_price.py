bread_price = [[0.5, 5], [0.6, 5.5], [1.1, 6.8], [1.4, 7]]


def BGD_step_gradient(w0_current, w1_current, points, learningRate):
    # w0_current w0系数值
    # w1_current w1系数值
    w0_gradient = 0
    w1_gradient = 0
    for i in range(len(points)):
        # points 特征值
        # costFunction equals to 0.5 * segema((y - y_prediction)^2)
        x = points[i][0]
        y = points[i][1]
        w0_gradient += 1.0 * (y - ((w1_current * x) + w0_current))  # 对w0求导
        w1_gradient += 1.0 * x * (y - ((w1_current * x) + w0_current))  # 对w1求导
    new_w0 = w0_current + (learningRate * w0_gradient)
    new_w1 = w1_current + (learningRate * w1_gradient)
    return [new_w0, new_w1]


def gradient_decent_runner(points, start_w0, start_w1, learningRate, run_iterations):
    # points 特征值
    # learningRate 学习率
    # run_iterations 迭代次数
    w0 = start_w0
    w1 = start_w1
    for i in range(run_iterations):
        w0, w1 = BGD_step_gradient(w0, w1, points, learningRate)
    return [w0, w1]


def predict(w0, w1, wheat):
    price = w1 * wheat + w0
    return price


if __name__ == '__main__':
    learning_rate = 0.1  # 学习率
    num_iter = 100000  # 迭代次数
    w0, w1 = gradient_decent_runner(points=bread_price,
                                    start_w0=1,
                                    start_w1=1,
                                    learningRate=learning_rate,
                                    run_iterations=num_iter)
    print(w0, w1)
    price = predict(w0, w1, 0.9)  # 预测值 当x为0.9时
    print("prices = ", price)
