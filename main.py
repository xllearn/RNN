# main.py
import torch
from d2l import torch as d2l
from config import batch_size, num_steps, num_hiddens, num_epochs, learning_rate
from model import RNNModelScratch, get_params, init_rnn_state, rnn
from train import train_ch8


def main():
    device = d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, learning_rate, num_epochs, device)
    d2l.plt.show()


if __name__ == "__main__":
    main()
