import torch
from tqdm.auto import tqdm
import numpy as np

def train_func(model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy, device):

  model.to(device)
  model.train()
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy(y_pred.argmax(dim=1), y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}")

  model.eval()
  with torch.inference_mode():
    test_loss, test_acc = 0, 0
    for X, y in test_dataloader:
      X, y = X.to(device), y.to(device)
      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      test_loss += loss
      test_acc += accuracy(y_pred.argmax(dim=1), y)

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}")

    return train_loss, train_acc, test_loss, test_acc
