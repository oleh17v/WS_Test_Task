from sklearn.ensemble import RandomForestClassifier
from MnistClassifierInterface import MnistClassifierInterface


class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, train_dataset):
        X_train, y_train = train_dataset
        self.clf.fit(X_train, y_train)

    def predict(self, test_dataset):
        X_test, _ = test_dataset
        return self.clf.predict(X_test)