import pandas as pd
import matplotlib
# Weird fix for getting graphs to show
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn.metrics as metrics


class Titanic():
    def __init__(self):
        # There was no skewed data found in the data files
        self.file = pd.read_csv("./data_files/Titanic/traintest.csv")
        self.raw_file = pd.read_csv("./data_files/Titanic/predict.csv")
        self.tested = pd.read_csv("./data_files/Titanic/tested.csv")

        self.file = self.prep_data(self.file)
        self.raw_file = self.prep_data(self.raw_file)
        self.tested = self.prep_data(self.tested)

        self.display_info(self.file)

        self.display_barchart_sex()
        self.display_barchart_class()
        plt.show()

        print("\n---------------Training Models---------------\n")
        self.models = self.train_models()

        print("---------------Testing accuracy---------------\n")
        for model in self.models:
            self.test_accuracy(model)

        # In conclusion, the SVM model with a linear kernal was the best model, since it predicted 99% correct againts the tested data. The biggest factor in survival was your Sex, followed by Class.


    def encode(self, data, to_convert):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(data[to_convert])
        encoded_values = self.le.transform(data[to_convert])
        data.drop([to_convert], axis=1, inplace=True)
        data[to_convert] = encoded_values
        return data

    def display_info(self, data_file):
        print("---------------Head---------------\n")
        print(data_file.head())
        print("\n---------------Info---------------\n")
        print(data_file.info())
        print("\n---------------Describe---------------\n")
        print(data_file.describe())

    def test_accuracy(self, model):
        print(f'Model: {model}')
        predict = model.predict(self.raw_file)
        print(
            f'Tested Accuracy: {metrics.accuracy_score(self.tested["Survived"], predict)}')
        matrix = metrics.confusion_matrix(self.tested["Survived"], predict)
        self.explain_confusion_matrix(matrix)
        
    def train_data(self, resp, test_size=0.75):
        y = self.file[resp]
        predictors = list(self.file.columns)
        predictors.remove(resp)
        x = self.file[predictors]
        return train_test_split(x, y, random_state=1111, test_size=test_size)

    def train_models(self):
        # For the logictic regression model, using 0.4 yielded the best results
        x_train, x_test, y_train, y_test = self.train_data("Survived", 0.4)
        lg_model = self.train_lg_model(x_train, x_test, y_train, y_test)

        x_train, x_test, y_train, y_test = self.train_data("Survived", 0.2)
        rf_model = self.train_random_forest_model(
            x_train, x_test, y_train, y_test)

        x_train, x_test, y_train, y_test = self.train_data("Survived", 0.4)
        svm_model = self.train_support_vector_machines_model(
            x_train, x_test, y_train, y_test)

        return lg_model, rf_model, svm_model

    def train_lg_model(self, x_train, x_test, y_train, y_test):
        # For the Logistic regression model, using the default values yielded the most accuracy
        lg = LogisticRegression()
        model = lg.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(
            f'Logistic Regression accuracy: {metrics.accuracy_score(y_test, y_pred)}')
        matrix = metrics.confusion_matrix(y_test, y_pred)
        self.explain_confusion_matrix(matrix)
        return model

    def train_random_forest_model(self, x_train, x_test, y_train, y_test):
        # For the Random classifier model, using the n_estimators of 1000 yielded the most accuracy
        clf = RandomForestClassifier(n_estimators=1000)
        model = clf.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(
            f'Random forest accuracy: {metrics.accuracy_score(y_test, y_pred)}')
        matrix = metrics.confusion_matrix(y_test, y_pred)
        self.explain_confusion_matrix(matrix)
        return model

    def train_support_vector_machines_model(self, x_train, x_test, y_train, y_test):
        # For the SVM model, using the linear kernal yielded the most accuracy. I chose this model because i could spell it out in google, and it was on the example slides for other models.
        clf=svm.SVC(kernel="linear")
        model=clf.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(f'SVM accuracy: {metrics.accuracy_score(y_test, y_pred)}')
        matrix = metrics.confusion_matrix(y_test, y_pred)
        self.explain_confusion_matrix(matrix)
        return model

    def get_survived_and_dead(self):
        return self.file[self.file['Survived'].isin([1])], self.file[self.file['Survived'].isin([0])]

    def explain_confusion_matrix(self, matrix):
        print(
            f'Confusion matrix:\n {matrix}')
        print(f'''
        Confusion matrix explained:
            Survived:
                - model got right: {matrix[0][0]}
                - model got wrong: {matrix[0][1]}
            Dead:
                - model got wrong: {matrix[1][0]}
                - model got right: {matrix[1][1]}
        ''')

    def prep_data(self, data):
        # The ticket column had to be dropped since it was not relevant what ticket number the person had
        data.drop(['Ticket'], axis=1, inplace=True)
        # The name column had to be dropped since it's irrelevant and not a numerical value
        data.drop(['Name'], axis=1, inplace=True)
        # The Cabin column had to be dropped because it's not a strict numerical value, and couldn't be computed
        data.drop(['Cabin'], axis=1, inplace=True)
        # The Embarked column had to be dropped because it's not relevant where the person embarked on
        data.drop(['Embarked'], axis=1, inplace=True)
        # The Age value in certain rows was filled where it wasn't provided, because you can't train the model with missing data.
        data.fillna({"Age": data["Age"].median()}, inplace=True)
        # The Fare value had to be filled where it wasn't provided, because you can't train the model with missing data.
        data.fillna({"Fare": data["Fare"].median()}, inplace=True)
        # The following function encodes Sex to 0/1, because you need numeric data to train your model.
        return self.encode(data, "Sex")

    def display_barchart_sex(self):
        # This chart shows us that though there was significantly more men than women on the Titanic, a much higher percentage of men died compared to women. For the men, close to 75% died, on the womens side, around 20% died. This is not suprising since back in the day, it was common to evacuate women and children first.

        survived, dead=self.get_survived_and_dead()

        survived_men=survived[survived["Sex"].isin([1])]
        survived_women=survived[survived["Sex"].isin([0])]
        dead_men=dead[dead["Sex"].isin([1])]
        dead_women=dead[dead["Sex"].isin([0])]

        fix, ax=plt.subplots()
        x=["Women", "Men"]
        y1=[len(survived_women), len(survived_men)]
        y2=[len(dead_women), len(dead_men)]
        ax.bar(x, y1)
        ax.bar(x, y2, bottom=y1, color='r')
        plt.legend(["Alive", "Dead"])
        plt.ylabel("Number of people")

    def display_barchart_class(self):
        # This chart shows that though the majority of passengers were 3rd class, they have the largest amount of people who died. Compared to the first and second class, where in 2nd class close to 50% died, and in first class less than 50% died. This shows us that the class you were in had a direct relationship with your likelyhood of survival.

        survived, dead=self.get_survived_and_dead()

        survived_first=survived[survived["Pclass"].isin([1])]
        survived_second=survived[survived["Pclass"].isin([2])]
        survived_third=survived[survived["Pclass"].isin([3])]
        dead_first=dead[dead["Pclass"].isin([1])]
        dead_second=dead[dead["Pclass"].isin([2])]
        dead_third=dead[dead["Pclass"].isin([3])]

        fix, ax=plt.subplots()
        x=["1st Class", "2nd Class", "3rd Class"]
        y1=[len(survived_first), len(survived_second), len(survived_third)]
        y2=[len(dead_first), len(dead_second), len(dead_third)]
        ax.bar(x, y1)
        ax.bar(x, y2, bottom=y1, color='r')
        plt.legend(["Alive", "Dead"])
        plt.ylabel("Number of people")


if __name__ == "__main__":
    T=Titanic()
