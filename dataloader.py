from vocab import Vocab
from dataset import GPT2Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



def collate_fn(inputs):

    src_inputs,trg_inputs = list(zip(*inputs))

    src_inputs = pad_sequence(src_inputs,batch_first=True,padding_value=0)
    trg_inputs = pad_sequence(trg_inputs,batch_first=True,padding_value=0)

    return [src_inputs,trg_inputs]


def get_loader(args):
    

    vocab = Vocab(args)
    train_dataset = GPT2Dataset(args,vocab.vocab_src,vocab.vocab_trg,args.train_json)
    test_dataset = GPT2Dataset(args,vocab.vocab_src,vocab.vocab_trg,args.test_json)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)

    return train_loader,test_loader



