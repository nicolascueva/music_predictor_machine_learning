import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music_dataset.csv')
X = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
#model.fit(X, y)
#predictions = model.predict([ [21, 1], [22, 0] ])
#print(predictions)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
predictions_whole = model.predict(X_test)

score = accuracy_score(y_test, predictions_whole)
print(score)


