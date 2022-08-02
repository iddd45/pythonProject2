# 感知机为一个二分类
class Perceptron(object):
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func
        # 权重向量初始化为0
        # 列表表示
        self.weights = [0.0 for _ in range(input_para_num)]

    def __str__(self):
        return "final weights\n\tw0 = {:.2f}\n\tw1 = {:.2f}\n\tw2={:.2f}".format(self.weights[0], self.weights[1],
                                                                                 self.weights[2])

    def predict(self, row_vec):
        # 权重相乘之和
        # 输入层
        act_value = 0.0
        for i in range(len(self.weights)):
            act_value += self.weights[i] * row_vec[i]
        return self.activator(act_value)

    def train(self, dataset, interation, rate):
        # dataset:数据集
        # interation:训练次数
        # rate:学习率
        for i in range(interation):
            for input_vec_label in dataset:
                # input_vec_label 标签
                prediction = self.predict(row_vec=input_vec_label)
                # 更新权重
                self._update_weights(input_vec_label, prediction, rate)

    def _update_weights(self, input_vec_label, prediction, rate):
        # input_vec_label:标签
        # prediction:预测值
        # rate:学习率
        delta = input_vec_label[-1] - prediction
        # 预测值与真实值的差值
        for i in range(len(self.weights)):
            # 更新每个权重
            self.weights[i] += rate * delta * input_vec_label[i]


# define activate function
def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0


# build datasets
def get_training_dataset():
    datasets = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    # [-1, 1, 1] -> 1, [-1, 0, 0] -> 0, [-1, 1, 0] -> 0, [-1, 0, 1] -> 0
    return datasets


def train_and_perception():
    p = Perceptron(3, func_activator)
    dataset = get_training_dataset()
    p.train(dataset, 10000, 0.1)
    return p


if __name__ == '__main__':
    # 训练and感知机
    and_perception = train_and_perception()
    print(and_perception)
