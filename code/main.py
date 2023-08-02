#!/usr/bin/env python

import os
import sys
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR


def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (recommended) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss),
                           list_of_unpadded_feature_length (for CTCLoss)
    """
    # === write your code here ===
    pass


def train(train_dataloader, model, ctc_loss, optimizer):
    # === write your code here ===
    pass


def decode():
    # === write your code here ===
    pass

def compute_accuracy():
    # === write your code here ===
    pass

def main():
    training_set = YOUR_TRAINING_SET
    test_set = YOUR_TEST_SET

    train_dataloader = TRAIN_DATALOADER
    test_dataloader = TEST_DATALOADER

    model = LSTM_ASR

    # your can simply import ctc_loss from torch.nn
    loss_function = CTC_LOSS_FUNCTION

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = YOUR_NUM_EPOCHS
    for epoch in range(num_epochs):
        train(train_dataloader, model, loss_function, optimizer)

    # Testing (totally by yourself)
    decode()

    # Evaluate (totally by yourself)
    compute_accuracy()


if __name__ == "__main__":
    main()
