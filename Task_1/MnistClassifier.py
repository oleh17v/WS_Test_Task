from models.RandomForestModel import RandomForestModel
from models.CNNMnistModel import CNNMnistModel
from models.FFNNMnistModel import FFNNMnistModel


# Unified classifier
class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'nn':
            self.model = FFNNMnistModel()
        elif algorithm == 'cnn':
            self.model = CNNMnistModel()
        else:
            raise ValueError("Wrong algorithm input. Algorithm must be 'rf', 'nn', or 'cnn'")

    def train(self, train_dataset, val_dataset=None):
        if isinstance(self.model, RandomForestModel):
            self.model.train(train_dataset)
        else:
            self.model.train(train_dataset, val_dataset)

    def predict(self, test_dataset):
        return self.model.predict(test_dataset)
