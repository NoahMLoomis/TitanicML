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
        self.display_info()
        self.display_charts()
        self.le = preprocessing.LabelEncoder()

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

    def display_charts(self):
        self.file['Survived'].value_counts().plot(kind="bar")
        fig, ax = plt.subplots()
        ax.bar(self.file['Sex'], self.file['Survived'])        
        # male =  self.file[ self.file["Male"].isin([1])]
        # female =  self.file[ self.file["Female"].isin([0])]
        

    def prep_data(self):
        self.file.drop(['Ticket'], axis=1, inplace=True)
        self.file.drop(['Name'], axis=1, inplace=True)
        self.file.drop(['Cabin'], axis=1, inplace=True)
        self.file.fillna({"Age": self.file["Age"].median()}, inplace=True)
        self.encode("Sex")


if __name__ == "__main__":
    T = Titanic()
