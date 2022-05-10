import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics


class Titanic():
    def __init__(self):
        self.file = pd.read_csv("./data_files/Titanic/traintest.csv")
        self.prep_data()
        # self.display_info()
        self.display_charts()
        self.le = preprocessing.LabelEncoder()
        x_train, x_test, y_train, y_test = self.train("Survived")
        self.model(x_train, x_test, y_train, y_test)
        plt.show()

    def encode(self, to_convert):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.file[to_convert])
        encoded_values = self.le.transform(self.file[to_convert])
        self.file.drop([to_convert], axis=1, inplace=True)
        self.file[to_convert] = encoded_values
        print(self.file[to_convert])

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
        
        
    def model(self, x_train, x_test, y_train, y_test ):
        knn = KNeighborsClassifier(7)
        print(x_train)
        model = knn.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # self.le.inverse_transform(y_pred)
        print(metrics.accuracy_score(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))
        
    def display_charts(self):
        survived = self.file[self.file['Survived'].isin([1])]
        # The following bar chart shows that more women survived than men.        
        survived["Sex"].value_counts().plot(kind="bar")
        
    def prep_data(self):
        self.file.drop(['Ticket'], axis=1, inplace=True)
        self.file.drop(['Name'], axis=1, inplace=True)
        self.file.drop(['Cabin'], axis=1, inplace=True)
        self.file.drop(['Embarked'], axis=1, inplace=True)
        self.file.fillna({"Age": self.file["Age"].median()}, inplace=True)
        self.encode("Sex")


if __name__ == "__main__":
    T = Titanic()
