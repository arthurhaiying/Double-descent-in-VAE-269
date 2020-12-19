import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name,interpolate
from model import SentenceVAE



def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    fname = "%sHS%i.txt" % (args.anneal_function, args.hidden_size)
    if args.skip:
        fname = "skip"+fname
    file = open(fname,'w')

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        skip=args.skip
    )
    model = SentenceVAE(**params)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)
        elif anneal_function == 'none':
            return 1.0

    def rstrip(pred):
        end = pred[-1]
        i = pred.size(0)
        while i >= 0:
            i = i - 1
            if pred[i] != end:
                break
        return pred[:i+2]


    def bleu_score(pred,target,ignore_index=0):
        pred = rstrip(pred)
        #print("pred: ", pred)
        #print("target: ", target)
        indices,counts = torch.unique(pred,return_counts=True)
        indices_true,counts_true = torch.unique(target,return_counts=True)
        index_dict = {i.item():c.item() for i,c in zip(indices,counts)}
        index_dict_true = {i.item():c.item() for i,c in zip(indices_true,counts_true)}
        sum,length = 0,0
        for index in index_dict:
            if index != ignore_index:
                length += index_dict[index]
                if index in index_dict_true:
                    count_clip = min(index_dict[index],index_dict_true[index])
                    sum += count_clip
        return float(sum)/length

    def bleu_loss_fn(logp, target, length):
        batch_size = logp.size(0)
        _, pred = torch.max(logp,dim=-1) # get output sentence with maximum likelihood
        target = target[:, :torch.max(length).item()].contiguous()
        #print("pred size: %s target size: %s" % (pred.size(),target.size()))
        loss = 0.0
        for pred_row,target_row in zip(pred,target):
            loss += bleu_score(pred_row,target_row,ignore_index=datasets['train'].pad_idx)
        return loss/batch_size
    '''
    def sample(mean,logv):
        sample_size = 100
        batch_size, latent_size = args.batch_size, args.latent_size
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size,latent_size,sample_size])
        z = z * std.unsqueeze(-1) + mean.unsqueeze(-1)
        return z
        '''


    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        #print("log p shape: %s" %(logp.size(),))
        #print("target shape: %s" %(batch['target'].size(),))
        #print("max length: %s" %torch.max(length).item())

        #bleu_loss = bleu_loss_fn(logp,target,length)
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        #print("resize log p shape: %s" %(logp.size(),))
        #print("resize target shape: %s" %(target.size(),))
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                #print("batch length: %s" %batch['length'])
                #print("batch first input size: %s" % (batch['input'].size(),))
                #print("batch first input length: %s" % torch.count_nonzero(batch['input'][0]).item())
                logp, mean, logv, z = model(batch['input'], batch['length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                bleu_loss = bleu_loss_fn(logp,batch['target'],batch['length'])
                #z_sample = sample(mean,logv)

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)
                tracker['KL'] = torch.cat((tracker['KL'], (KL_loss/batch_size).data.view(1,-1)), dim=0)
                tracker['BLEU'] = torch.cat((tracker['BLEU'], torch.FloatTensor([bleu_loss])))
                #tracker['z_sample'] = torch.cat((tracker['z_sample'], z_sample), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, BLEU-score %6.5f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                          KL_loss.item()/batch_size, KL_weight,bleu_loss))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                                        pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            #print("z_sample size: %s" %(tracker['z_sample'].size(),))
            #print("z0 %s" % tracker['z_sample'][0][0].data)
            file.write("%s Epoch %02d/%i, Mean ELBO %9.4f, Mean KL %9.4f, Mean BLEU %6.4f\n" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean(),tracker['KL'].mean(),tracker['BLEU'].mean()))
            file.flush()

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)

    file.close()


if __name__ == '__main__':
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

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear','none']
    assert 0 <= args.word_dropout <= 1
    main(args)
