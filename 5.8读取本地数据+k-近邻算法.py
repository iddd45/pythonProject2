import csv
import operator
import random
import math


def loadDatasets(filename, split, trainingSet=[], testSet=[]):
    # split选取部分作为训练集，另外一部分作为测试集
    with open(filename, 'r',) as csvfile:
        lines = csv.reader(csvfile)
        datasets = list(lines)
        for x in range(len(datasets) - 1):
            for y in range(4):
                datasets[x][y] = float(datasets[x][y])
            if random.random() < split:
                trainingSet.append(datasets[x])
            else:
                testSet.append(datasets[x])


# 计算前两个特征样本之间的距离
def EuclidDist(instance1, instance2, length):
    # length为向量维度
    distance = 0.0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainSet, testInstance, k):
    # 对于测试样本选取k个剧里最小的值
    # 返回在测试集中欧式剧里最小的点
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainSet)):
        dist = EuclidDist(testInstance, trainSet[x], length)
        # 将第x个训练元素，以及其对应的欧式距离存入distance中
        distance.append((trainSet[x], dist))
    # operator.itemgetter(1)为第1个位置的参数的值
    # 以第一个位置的值排序，及dist
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors


# trainSet = [[3, 2, 6, 'a'], [1, 2, 4, 'b'], [2, 2, 2, 'b'], [1, 5, 4, 'a']]
# testInstance = [4, 5, 6]
# k = 1
# neighbors = getNeighbors(trainSet, testInstance, k)
# print(neighbors)


def getClass(neighbors):
    # 分类器
    # 创建一个字典类型
    classVotes = {}
    for x in range(len(neighbors)):
        instance_class = neighbors[x][-1]
        # 统计得分最高的
        if instance_class in classVotes:
            classVotes[instance_class] += 1
        else:
            classVotes[instance_class] = 1
    # sort应用于列表，sorted应用于所有可迭代的对象
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    # 对于预测值以及测试值
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.7
    loadDatasets('datasets/iris.data', split, trainingSet, testSet)
    print("训练集合：" + repr(len(trainingSet)))
    print("测试结合：" + repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # 找取最近3个
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getClass(neighbors)
        predictions.append(result)
        print(">预测=" + repr(result) + ',实际=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('精度为：' + repr(accuracy) + '%')


main()

# k值邻近为在k范围内测试值对于样本值最近的距离，加权投票给出预测值
