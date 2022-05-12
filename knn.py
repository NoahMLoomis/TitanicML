import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics


class Titanic():
    def __init__(self):
        self.file = pd.read_csv("./data_files/Titanic/traintest.csv")
        self.raw_file = pd.read_csv("./data_files/Titanic/predict.csv")
        self.file = self.prep_data(self.file)
        # self.display_info()
        self.display_barchart_sex()
        self.display_barchart_class()
        self.le = preprocessing.LabelEncoder()
        x_train, x_test, y_train, y_test = self.train("Survived")
        self.model = self.train_model(x_train, x_test, y_train, y_test)

        # self.raw_file = self.prep_data(self.raw_file)
        # self.le = preprocessing.LabelEncoder()
        # print(self.raw_file)
        # y_pred2 = self.model.predict(self.raw_file)
        # print(metrics.accuracy_score(self.raw_file, y_pred2))

        plt.show()

    def encode(self, data, to_convert):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(data[to_convert])
        encoded_values = self.le.transform(data[to_convert])
        data.drop([to_convert], axis=1, inplace=True)
        data[to_convert] = encoded_values
        return data

    def display_info(self):
        print(self.file.head())
        print(self.file.info())
        print(self.file.describe())

    def train(self, resp):
        y = self.file[resp]
        predictors = list(self.file.columns)
        predictors.remove(resp)
        x = self.file[predictors]
        return train_test_split(x, y, random_state=1111)

    def train_model(self, x_train, x_test, y_train, y_test):
        knn = KNeighborsClassifier(7)
        model = knn.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(metrics.accuracy_score(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))
        return model

    def display_barchart_sex(self):
        survived, dead = self.get_survived_and_dead()
        survived_men = survived[survived["Sex"].isin([1])]
        survived_women = survived[survived["Sex"].isin([0])]

        dead_men = dead[dead["Sex"].isin([1])]
        dead_women = dead[dead["Sex"].isin([0])]

        fix, ax = plt.subplots()
        x = ["Women", "Men"]
        y1 = [len(survived_women), len(survived_men)]
        y2 = [len(dead_women), len(dead_men)]
        ax.bar(x, y1)
        ax.bar(x, y2, bottom=y1, color='r')
        plt.legend(["Alive", "Dead"])
        plt.ylabel("Number of people")

    def display_barchart_class(self):
        survived, dead = self.get_survived_and_dead()

        survived_first = survived[survived["Pclass"].isin([1])]
        survived_second = survived[survived["Pclass"].isin([2])]
        survived_third = survived[survived["Pclass"].isin([3])]

        dead_first = dead[dead["Pclass"].isin([1])]
        dead_second = dead[dead["Pclass"].isin([2])]
        dead_third = dead[dead["Pclass"].isin([3])]

        fix, ax = plt.subplots()
        x = ["1st Class", "2nd Class", "3rd Class"]
        y1 = [len(survived_first), len(survived_second), len(survived_third)]
        y2 = [len(dead_first), len(dead_second), len(dead_third)]
        ax.bar(x, y1)
        ax.bar(x, y2, bottom=y1, color='r')
        plt.legend(["Alive", "Dead"])
        plt.ylabel("Number of people")

    def get_survived_and_dead(self):
        return self.file[self.file['Survived'].isin([1])], self.file[self.file['Survived'].isin([0])]

    def prep_data(self, data):
        data.drop(['Ticket'], axis=1, inplace=True)
        data.drop(['Name'], axis=1, inplace=True)
        data.drop(['Cabin'], axis=1, inplace=True)
        data.drop(['Embarked'], axis=1, inplace=True)
        data.fillna({"Age": data["Age"].median()}, inplace=True)
        return self.encode(data, "Sex")


if __name__ == "__main__":
    T = Titanic()
