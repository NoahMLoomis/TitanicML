import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier

chess = pd.read_csv("./data_files/kingrookchess.csv")

# print(chess.head())
# print(chess.describe())
# print(chess.info())
# There is a regression, from 14 to 1, telling us theres much more data with 14 moves than 1 move
chess[['endgame_moves']].value_counts().plot(kind='bar')

le = preprocessing.LabelEncoder()

le.fit(chess['w_king_file'])
encoded_w_king_file = le.transform(chess["w_king_file"])
chess.drop(['w_king_file'], axis=1, inplace=True)
chess["w_king_file"] = encoded_w_king_file

le = preprocessing.LabelEncoder()

le.fit(chess['w_rook_file'])
encoded_w_rook_file = le.transform(chess["w_rook_file"])
chess.drop(['w_rook_file'], axis=1, inplace=True)
chess["w_rook_file"] = encoded_w_rook_file

le = preprocessing.LabelEncoder()

le.fit(chess['b_king_file'])
encoded_b_king_file = le.transform(chess["b_king_file"])
chess.drop(['b_king_file'], axis=1, inplace=True)
chess["b_king_file"] = encoded_b_king_file
# print(list(le.classes_))

plt.show()

resp = "endgame_moves"
y = chess[resp]

predictors = list(chess.columns)
predictors.remove(resp)
x = chess[predictors]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1111, test_size=0.04)

# 3 = 61%
# 5 = 72%
# 7 = 77%
knn = KNeighborsClassifier(7)
model = knn.fit(x_train, y_train)

y_pred = model.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred))
# print(metrics.confusion_matrix(y_test, y_pred))

other_df = pd.DataFrame([{"w_king_file": 1,"w_king_rank": 1,"w_rook_file": 4,"w_rook_rank": 6,"b_king_file": 4,"b_king_rank": 7 }])
# print(model.predict(other_df))

raw_chess = pd.read_csv("./data_files/raw_kingrookchess.csv")
raw_actual = pd.read_csv("./data_files/raw_kingrookchess_actual.csv")

le = preprocessing.LabelEncoder()

le.fit(raw_chess['w_king_file'])
encoded_w_king_file = le.transform(raw_chess["w_king_file"])
raw_chess.drop(['w_king_file'], axis=1, inplace=True)
raw_chess["w_king_file"] = encoded_w_king_file

le.fit(raw_chess['w_rook_file'])
encoded_w_rook_file = le.transform(raw_chess["w_rook_file"])
raw_chess.drop(['w_rook_file'], axis=1, inplace=True)
raw_chess["w_rook_file"] = encoded_w_rook_file


le.fit(raw_chess['b_king_file'])
encoded_b_king_file = le.transform(raw_chess["b_king_file"])
raw_chess.drop(['b_king_file'], axis=1, inplace=True)
raw_chess["b_king_file"] = encoded_b_king_file

print(f'raw chess: {raw_chess}')
y_pred2 = model.predict(raw_chess)
# print(metrics.accuracy_score(raw_actual, y_pred2))
