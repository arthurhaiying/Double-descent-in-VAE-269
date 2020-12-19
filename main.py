import argparse
from multiprocessing import Process
import sys

from train import main


NUM_EPOCH = 10
HIDDEN_SIZES = [1] + [2*(i+1) for i in range(15)]
ANNEAL = "logistic"
USE_SKIP = True

def do_main(hidden_size):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')
    parser.add_argument('--skip',action='store_true')

    arg_list = ['--test', f'-ep={NUM_EPOCH}', f'-hs={hidden_size}', f'-bin=bin/HS{hidden_size}',f'-af={ANNEAL}']
    if USE_SKIP:
        arg_list.append("--skip")
    args = parser.parse_args(arg_list)
    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear','none']
    assert 0 <= args.word_dropout <= 1
    print(args)
    main(args)


if __name__ == '__main__':
    processes = []
    for hs in HIDDEN_SIZES:
        p = Process(target=do_main,args=(hs,))
        processes.append(p)
    for p in processes:
        p.start()
    for i,p in enumerate(processes):
        p.join()
        print("Finish experiment %i" %i)


