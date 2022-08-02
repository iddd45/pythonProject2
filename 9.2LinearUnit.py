# 利用SGD(随机梯度下降)解决线性回归问题
from csv import reader


class LinearUnit(object):
    def __init__(self, input_para_num, acti_func):
        # input_para_num
        # acti_func 激活函数
        self.activator = acti_func

    def predict(self, row_vec):
        act_values = self.weights[0]
        for i in range(len(row_vec) - 1):
            # 标题减一行
            act_values += self.weights[i + 1] * row_vec[i]
        return self.activator(act_values)

    def train_sgd(self, dataset, rate, n_epoch):
        # 训练权值
        # dataset 数据集
        # rate 学习率
        # n_epoch 迭代次数
        self.weights = [0.0 for i in range(len(dataset))]  # 权重向量初始化为0
        for i in range(n_epoch):
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                self.update_weights(input_vec_label, prediction, rate)  # 更新权重

    def update_weights(self, input_vec_label, prediction, rate):
        # 更新权值
        delta = input_vec_label[-1] - prediction
        # 更新哑元的权值
        self.weights[0] = rate * delta + self.weights[0]
        # 更新其余权值
        for i in range(len(self.weights)):
            self.weights[i + 1] = self.weights[i + 1] + rate * delta * input_vec_label[i]


def func_activator(input_value):
    # 定义激活函数
    return input_value


class Database():
    def __init__(self):
        self.dataset = list()

    def load_csv(self, filename):
        # 导入csv文件
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            # 读取表头
            heading = next(csv_reader)
            # 读取数据
            for row in csv_reader:
                if not row:  # 判断是否有空行
                    continue
                self.dataset.append(row)

    def dataset_str_to_float(self):
        col_len = len(self.dataset[0])
        for row in self.dataset:
            for column in range(col_len):
                row[column] = float(row[column].strip())

    def _dataset_minmax(self):  # 定义寻找极值的私有方法
        self.minmax = list()
        for i in range(len(self.dataset[0])):
            col_values = [row[i] for row in self.dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            self.minmax.append([value_min, value_max])

    def normalize_dataset(self):
        # 标准化
        self._dataset_minmax()
        for row in self.dataset:
            for i in range(len(row)):
                row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
        return self.dataset


def get_traing_dataset():
    db = Database()  # 构建训练集
    db.load_csv("winequality-white.csv")
    db.dataset_str_to_float()
    dataset = db.normalize_dataset()
    return dataset


def train_linear_unit():
    dataset = get_traing_dataset()
    learning_rate = 0.1  # 学习率
    num_iter = 100  # 迭代次数
    # 创建训练线性单元
    linear_unit = LinearUnit(len(dataset[0]), func_activator)
    # 训练
    linear_unit.train_sgd(dataset, learning_rate, num_iter)
    # 返回训练好的线性单元
    return linear_unit


if __name__ == '__main__':
    linear_unit = train_linear_unit()
    # prediction = linear_unit.predict("测试向量")
