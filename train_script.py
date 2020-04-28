import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modeles_library import my_lstm
import random
import os


def train(args):
    torch.use_cuda = True
    cuda = torch.cuda.is_available()
    print("cuda:", cuda)

    X_train = np.load(os.path.join(args.data_dir, args.X_train_file))
    y_train = np.load(os.path.join(args.data_dir, args.y_train_file))
    X_test = np.load(os.path.join(args.data_dir, args.X_test_file))
    y_test = np.load(os.path.join(args.data_dir, args.y_test_file))

    embedding_size = 1
    hidden_dim = 10
    target_size = 1
    num_layers = 1

    model = my_lstm(embedding_size, hidden_dim, target_size, num_layers=num_layers)
    loss_function = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if cuda:
        model.cuda()

    for epoch in range(2):
        train_set = list(zip(X_train, y_train))
        random.shuffle(train_set)
        for i, (seq, label) in enumerate(train_set):
            model.zero_grad()
            if cuda:
                t_seq = torch.from_numpy(seq).cuda()
                t_label = torch.from_numpy(np.array(label)).cuda()
            else:
                t_seq = torch.from_numpy(seq)
                t_label = torch.from_numpy(np.array(label))
            output = model(t_seq)

            loss = loss_function(output, t_label)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                error = 0
                for k in range(1000):
                    j = np.random.randint(0, len(y_test))
                    if cuda:
                        t_seq_test = torch.from_numpy(X_test[j]).cuda()
                    else:
                        t_seq_test = torch.from_numpy(X_test[j])
                    output = model(t_seq_test)
                    error += np.abs(y_test[j] - output.cpu().detach().numpy())

                print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--try_arg', type=int, default=64, metavar='N',
                        help='arg for testing parser')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_STOCKDATA'),
                        help='data directory path')
    parser.add_argument('--X_train_file', type=str) # no default to test argument are well given
    parser.add_argument('--y_train_file', type=str, default='X_train.npy')
    parser.add_argument('--X_test_file', type=str, default='X_test.npy')
    parser.add_argument('--y_test_file', type=str, default='y_test.npy')
    train(parser.parse_args())
