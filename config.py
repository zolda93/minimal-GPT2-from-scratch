import argparse

def load_args():

    parser = argparse.ArgumentParser('GPT2')

    # vocab config

    parser.add_argument('--src_corpus',type=str,default='src_corpus.txt')
    parser.add_argument('--trg_corpus',type=str,default='trg_corpus.txt')

    parser.add_argument('--src_prefix',type=str,default='nmt_src_vocab')
    parser.add_argument('--trg_prefix',type=str,default='nmt_trg_vocab')

    parser.add_argument('--vocab_src_file',type=str,default='./nmt_src_vocab.model')
    parser.add_argument('--vocab_trg_file',type=str,default='./nmt_trg_vocab.model')

    parser.add_argument('--train_dataset',type=str,default='./train_dataset.txt')
    parser.add_argument('--test_dataset' ,type=str,default='./test_dataset.txt')

    parser.add_argument('--train_json',type=str,default='./train_json.json')
    parser.add_argument('--test_json',type=str,default='./test_json.json')
    

    # model
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--embedding_dim',type=int,default=256)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--max_len',type=int,default=100)
    parser.add_argument('--N',type=int,default=6)

    # training
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--cuda',type=bool,default=True)
    parser.add_argument('--epochs',type=int,default=20)

    args,unknown = parser.parse_known_args()
    return args
    

