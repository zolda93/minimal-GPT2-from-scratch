import os
import csv
import re
import shutil
import random
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import urllib3
import zipfile
import sentencepiece as spm
from sklearn.model_selection import train_test_split


http = urllib3.PoolManager()
url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
filename = 'cornell_movie_dialogs_corpus.zip'
zipfilename = os.path.join(os.getcwd(),filename)
path_to_movie_lines = '{}/cornell movie-dialogs corpus/movie_lines.txt'.format(os.getcwd())
path_to_movie_conversations = '{}/cornell movie-dialogs corpus/movie_conversations.txt'.format(os.getcwd())



class Vocab:

    def __init__(self,args):

        self.corpus_src = args.src_corpus
        self.corpus_trg = args.trg_corpus
        self.prefix_src = args.src_prefix
        self.prefix_trg = args.trg_prefix
        self.vocab_src_file = args.vocab_src_file
        self.vocab_trg_file = args.vocab_trg_file
        self.ratings_train = args.train_dataset
        self.ratings_test = args.test_dataset
        
        self.vocab_size = 4000

        print('Start Downloading data ...')

        with http.request('GET',url,preload_content=False) as r,open(zipfilename,'wb') as out_file:
                shutil.copyfileobj(r,out_file)

        
        print('Extracting data...')

        with zipfile.ZipFile(zipfilename,'r') as zip_file:
            zip_file.extractall(os.getcwd())


        self.df = self.load_data()

        train_df,test_df = train_test_split(self.df,test_size=0.2)

        train_df.to_csv(self.ratings_train,sep='\t',index=False)
        test_df.to_csv(self.ratings_test,sep='\t',index=False)


        with open(self.corpus_src,'w',encoding='utf8') as f:
            f.write('\n'.join(self.df['SRC']))
        with open(self.corpus_trg,'w',encoding='utf8') as f:
            f.write('\n'.join(self.df['TRG']))

        self.build_vocab()

        self.prepare_data(self.ratings_train,args.train_json)
        self.prepare_data(self.ratings_test,args.test_json)



    def preprocess_sentence(self,sentence):

        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # removing contractions
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"he's", "he is", sentence)
        sentence = re.sub(r"she's", "she is", sentence)
        sentence = re.sub(r"it's", "it is", sentence)
        sentence = re.sub(r"that's", "that is", sentence)
        sentence = re.sub(r"what's", "that is", sentence)
        sentence = re.sub(r"where's", "where is", sentence)
        sentence = re.sub(r"how's", "how is", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can't", "cannot", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"n'", "ng", sentence)
        sentence = re.sub(r"'bout", "about", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        return sentence


    def load_data(self):

        print('Start loading data...')

        id2line = {}

        with open(path_to_movie_lines,errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            chunks = line.replace('\n','').split(' +++$+++ ')
            id2line[chunks[0]] = chunks[4]
        
        raw_src,raw_trg = [],[]

        with open(path_to_movie_conversations,'r') as f:
            lines = f.readlines()

        for line in lines:
            chunks = line.replace('\n','').split(' +++$+++ ')
            conversation = [chunk[1:-1] for chunk in chunks[3][1:-1].split(', ')]

            for i in range(len(conversation)-1):
                raw_src.append(self.preprocess_sentence(id2line[conversation[i]]))
                raw_trg.append(self.preprocess_sentence(id2line[conversation[i+1]]))

        print('Finish loading data...')

        df1 = pd.DataFrame(raw_src)
        df2 = pd.DataFrame(raw_trg)

        df1.rename(columns={0:'SRC'},errors='raise',inplace=True)
        df2.rename(columns={0:'TRG'},errors='raise',inplace=True)

        df = pd.concat([df1,df2],axis=1)
        df['src_len'] = ''
        df['trg_len'] = ''

        for idx in range(len(df['SRC'])):
            src_txt,trg_txt = str(df.iloc[idx]['SRC']),str(df.iloc[idx]['TRG'])
            res_src,res_trg = len(src_txt.split()),len(trg_txt.split())
            df.at[idx,'src_len'],df.at[idx,'trg_len'] = int(res_src),int(res_trg)

        df = df.drop_duplicates(subset=['SRC'])
        df = df.drop_duplicates(subset=['TRG'])

        # Filter the data that meets the condition and store it in a new variable.

        condition = (7 < df['src_len']) & (df['src_len'] <= 17) & (7 < df['trg_len']) & (df['trg_len'] <=17)

        df = df[condition]
        df = df.sample(n=1024*10,random_state=1234)
        return df


    def build_vocab(self):

        print('Start building vocab...')
        spm.SentencePieceTrainer.train(
            f"--input={self.corpus_src} --model_prefix={self.prefix_src} --vocab_size={self.vocab_size + 7}" +
            " --model_type=bpe" +
            " --max_sentence_length=999999" +               
            " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
            " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
            " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
            " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
            " --user_defined_symbols=[SEP],[CLS],[MASK]")


        spm.SentencePieceTrainer.train(
            f"--input={self.corpus_trg} --model_prefix={self.prefix_trg} --vocab_size={self.vocab_size + 7}" +
            " --model_type=bpe" +
            " --max_sentence_length=999999" +   
            " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
            " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
            " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
            " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
            " --user_defined_symbols=[SEP],[CLS],[MASK]")

        self.vocab_src = spm.SentencePieceProcessor()
        self.vocab_src.load(self.vocab_src_file)

        self.vocab_trg = spm.SentencePieceProcessor()
        self.vocab_trg.load(self.vocab_trg_file)


    def prepare_data(self,infile,outfile):

        print('Begin preparing data...')

        df = pd.read_csv(infile,sep='\t',engine='python')

        with open(outfile,'w') as f:
            for idx,row in df.iterrows():
                src_doc = row['SRC']
                if type(src_doc) != str:
                    continue
                tmp_src_sent = self.vocab_src.encode_as_pieces(src_doc)
                if len(tmp_src_sent) > 256:
                    tmp_src_sent = tmp_src_sent[:256]

                trg_doc = row['TRG']
                if type(trg_doc) != str:
                    continue
                tmp_trg_sent = self.vocab_trg.encode_as_pieces(trg_doc)
                if len(tmp_trg_sent) > 256:
                    tmp_trg_sent = tmp_trg_sent[:256]

                instance = {'SRC':tmp_src_sent,'TRG':tmp_trg_sent}
                f.write(json.dumps(instance))
                f.write('\n')



