"""ResNet18 model architecutre, training, and testing functions for CIFAR100."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        """KL loss forward."""
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss

def test(  # pylint: disable=too-many-locals
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float, List[float]]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float, List[float]]
        The loss and the accuracy of the global model
        and the list of accuracy for each classifier on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    correct_single = [0] * 4  # accuracy of each classifier within model
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output_lst = net(images)

            # ensemble classfiers' output
            ensemble_output = torch.stack(output_lst, dim=2)
            ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_lst)

            loss += criterion(ensemble_output, labels).item()
            _, predicted = torch.max(ensemble_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, single in enumerate(output_lst):
                _, predicted = torch.max(single, 1)
                correct_single[i] += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    accuracy_single = [correct / total for correct in correct_single]
    return loss, accuracy, accuracy_single