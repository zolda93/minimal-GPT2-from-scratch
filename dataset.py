import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm




class GPT2Dataset(Dataset):

    def __init__(self,args,vocab_src,vocab_trg,infile):
        
        #self.vocab = Vocab()
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_src
        self.src_sentences = []
        self.trg_sentences = []
        self.device = 'cuda' if args.cuda else 'cpu'

        line_cnt = 0
        with open(infile,'r') as f:
            for line in f:
                line_cnt += 1

        with open(infile,'r') as f:

            for i,line in enumerate(tqdm(f,total=line_cnt,desc=f"Loading{infile}",unit="Lines")):
                data = json.loads(line)
                src_sentence = [self.vocab_src.piece_to_id("[BOS]")] + [self.vocab_src.piece_to_id(p) for p in data['SRC']] + [self.vocab_src.piece_to_id("[EOS]")]
                trg_sentence = []
                trg_sentence = [0]*(len(src_sentence))

                for _ in range(args.max_len - len(src_sentence)):
                    src_sentence.append(self.vocab_src.piece_to_id("[PAD]"))

                trg_sentence += [self.vocab_trg.piece_to_id(p) for p in data["TRG"]] + [self.vocab_trg.piece_to_id("[EOS]")]

                for _ in range(args.max_len - len(trg_sentence)):
                    trg_sentence.append(self.vocab_trg.piece_to_id("[PAD]"))

                self.src_sentences.append(src_sentence)
                self.trg_sentences.append(trg_sentence)

    def __len__(self):
        assert len(self.src_sentences) == len(self.trg_sentences)
        return len(self.src_sentences)


    def __getitem__(self,idx):
        return (torch.tensor(self.src_sentences[idx]).to(self.device),torch.tensor(self.trg_sentences[idx]).to(self.device))



