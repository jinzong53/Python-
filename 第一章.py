import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.datasets import load_iris #从sklearn.datasets导入鸢尾花
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris() #获取鸢尾花集
x_train,x_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
knn = KNeighborsClassifier(n_neighbors=1) #得到knn对象


# 打鸢尾花中的数据
def printInfo():
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
    # print(iris_dataset['DESCR'] + "\n") 获取该数据集的描述
    print("该鸢尾花的种类：{}".format(iris_dataset['target_names']))
    print("该鸢尾花的各数据的名称：{}".format(iris_dataset['feature_names']))
    print("该鸢尾花的data的类型：{}".format(type(iris_dataset['data'])))
    print("target 是一维数组，每朵花对应其中一个数据:{}(行，列)".format(iris_dataset['target'].shape))
    print("Target:\n{}".format(
        iris_dataset['target']))  # 上述数字的代表含义由 iris['target_names'] 数组给出：0 代表 setosa，1 代表 versicolor， 2 代表 virginica。   #D  ##DA

# 打印切分数据集
def splitIris():
    print("x训练集data{}".format(x_train.shape))
    print("y训练集target{}".format(x_train.shape))
    print("x测试集data{}".format(x_test.shape))
    print("y测试集data{}".format(y_test.shape))

# 打印相关矩阵图
def printGragh():
    # 利用X_train中的数据创建DataFrame # 利用iris_dataset.feature_names中的字符串对数据列进行标记
    iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
    # 利用DataFrame创建散点图矩阵，按y_train着色
    grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
    plt.show() #打印图片

# 算法模型
def knnModle():
    knn.fit(x_train, y_train)

# 预测鸢尾花的的种类我们在野外发现了一朵鸢尾花，花萼长5cm 宽 2.9cm，花瓣长1cm 宽 0.2cm
def predict():
    x_newData = np.array([[5,2.9,1,0.2]]) # 新发现的鸢尾花的数据
    prediction = knn.predict(x_newData)
    print("prediction的类型是：{}".format(type(prediction)))
    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# 模型评估
def evaluation():
    y_pred = knn.predict(x_test)
    print("模型的预测结果: {}".format(y_pred))
    print("测试集实际结果: {}".format(y_test))
    print("相似度: {:.2f}".format(np.mean(y_pred == y_test))) # 在这里返回的条件成立的比例条件
if __name__ == '__main__':
    splitIris()
    printGragh()
    knnModle()
    predict()
    evaluation()