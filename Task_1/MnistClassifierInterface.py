from abc import ABC, abstractmethod


# Define the interface using ABC lib
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, train_dataset, val_dataset=None):
        pass

    @abstractmethod
    def predict(self, test_dataset):
        pass
