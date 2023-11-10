from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel as LMSingle
from gensim.models.coherencemodel import CoherenceModel
import argparse
import pandas as pd
import random
import gensim
import json
import os
from statistics import mean
import pickle

stemmer = SnowballStemmer("english")

def filter_pos(toks, pos1, pos2, select_pos):
    if len(select_pos)==0:
        return toks
    tokens = []
    for i in range(len(toks)):
        if (pos1[i] in select_pos) or (pos2[i] in select_pos):
            tokens.append(toks[i])
    return tokens 

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stem_ = stemmer.stem(item)
        stemmed.append(stem_)
    return stemmed

def get_random_number():
    return random.randint(0,50000)

class LDADocument:
    def __init__(self, app, id, body):
        self.app = app
        self.id = id
        self.body = body

class LDA_Model:
    def __init__(self, training_data=None, num_topics=10, pos='ALL', datadir='Preprocessed', results=None,
                 translated = False, use_multicore=True, core=24, iterations=100, save_corpus=False):
        self.num_topics = num_topics
        self.pos = pos
        self.translated = translated
        self.datadir = datadir
        self.use_multicore=use_multicore
        self.workers=core
        self.iterations=iterations
        self.results = results
        self.save_corpus = save_corpus

        if self.datadir.rfind('/')==-1:
            self.location = self.datadir
        elif self.datadir.rfind('/')==(len(self.datadir)-1):
            self.location = self.datadir[:-1]
        else:
            self.location = self.datadir[self.datadir.rfind('/')+1:]

        if (training_data is None):
            self.training_data = self.read_training_data()
        else:
            self.training_data = training_data

        self.models = self.create_model_from_training_data()

        scores = []
        outIdx = results.shape[0]
        models_best = []
        seed = 0
        for i in range(len(num_topics)):
            maxScore = 0
            nt = num_topics[i]
            print('With num_topics:', nt)
            coherence=self.compute_coherence(i)
            score = [coherence]
            models_best.append(self.models[i])
            for j in range(10):
                self.models[i]=self.prepare_model(nt)
                coherence=self.compute_coherence(i)
                score.append(coherence)
                print(j, '---------', coherence)
                print('SEED:',self.seed)
                if coherence>maxScore:
                    maxScore = coherence
                    models_best[i] = self.models[i]
                    seed = self.seed
            meanScore = mean(score)
            print('Coherence Score Max:', maxScore, '|Mean:', meanScore)
            self.results.loc[outIdx] = [self.location, nt, ' '.join(self.pos), self.translated, seed, maxScore, meanScore]
            outIdx+=1
            print(self.results)
        self.models = models_best
        

    def get_model(self):
        return self.model

    def create_model_from_training_data(self):
        document_ids = []
        token_collection = []
        for document in self.training_data:
            document_ids.append(document.id)
            token_collection.append(stem_tokens(document.body))

        self.document_ids = document_ids
        self.token_collection = token_collection

        self.dictionary = corpora.Dictionary(self.token_collection)
        #self.dictionary.filter_extremes(no_below=20, no_above=0.8)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.token_collection]

        self.models = []
        for nt in self.num_topics:
            self.models.append(self.prepare_model(nt))
        return self.models

    def prepare_model(self, topics_count):
        self.seed = get_random_number()
        if(self.use_multicore):
            ldamodel = LdaMulticore(self.corpus, num_topics=topics_count, id2word=self.dictionary,
                                passes=50, workers=self.workers, alpha='symmetric', random_state=self.seed,
                                    eta='auto', iterations=self.iterations)
            return ldamodel
        else:
            ldamodel = LMSingle(corpus =self.corpus, num_topics=topics_count, id2word=self.dictionary,
                                random_state=self.seed, passes=50, alpha='auto', eta='auto', iterations=self.iterations)
            return ldamodel

    def compute_coherence(self, idx):
        coherencemodel = CoherenceModel(model=self.models[idx], dictionary=self.dictionary, texts=self.token_collection, topn=10,
                                            coherence='c_v')
        value = coherencemodel.get_coherence()
        return value

    def read_training_data(self):
        print('----', len(os.listdir(self.datadir)), 'apps------')
        model_data = []
        for file in os.listdir(self.datadir):
            print(file[:-5])
            df = pd.read_excel(os.path.join(self.datadir, file))
            nrows = df.shape[0]
            for i in range(nrows):
                if (not self.translated) and df.loc[i]["isEnglish"]==0:
                    continue
                text = df.loc[i]["preprocessedLDA"]
                if not isinstance(text, str) or len(text)==0:
                    continue
                review = filter_pos(text.split(), df.loc[i]["POS1"].split(), df.loc[i]["POS2"].split(), self.pos)
                if len(review)>0:                
                    doc = LDADocument(file[:-5], df.loc[i]['reviewId'], review)
                    model_data.append(doc)
        print(len(model_data))
        if self.save_corpus:
            if not os.path.isdir('Corpus'):
                os.mkdir('Corpus')
            fname = self.location+'_'+'_'.join(self.pos)+'.corpus'
            with open('Corpus/'+fname, 'wb') as outfile:
                pickle.dump(model_data, outfile)
        return model_data

    def save_results(self, name, dest):
        self.results.to_csv(name)
        length = len(self.models)
        start = self.results.shape[0]-length
        for i in range(length):
            fname = os.path.join(dest, str(start+i)+'.model')
            self.models[i].save(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA Model')

    parser.add_argument('--data', type=str,
                        help='Data Directory', default="Data/Europe")

    parser.add_argument('--out', type=str,
                        help='Output Directory', default="Out")

    parser.add_argument('--result', type=str,
                        help='Results file', default="results.csv") 

    parser.add_argument('--config', type=str,
                    help='Configuration File', default="config.json")

    parser.add_argument('--multicore', type=bool,
                        help='Multicore', default=True)
    
    parser.add_argument('--core', type=int,
                        help='CPU Threads', default=8)
    
    parser.add_argument('--iteration', type=int,
                        help='Number of iterations', default=100)

    parser.add_argument('--corpus', type=str,
                    help='Corpus File', default="")

    parser.add_argument('--save_corpus', type=bool,
                        help='Save the corpus for this config?', default=False)

    args = parser.parse_args()
    if not(len(args.corpus) or len(args.data)):
        print("No training data or data directory")
        exit(1)
    datadir = args.data
    outdir = args.out
    multi_core=args.multicore
    num_core=args.core
    iters=args.iteration
    save_corpus = args.save_corpus

    training_data = None
    if len(args.corpus):
    	with open(args.corpus, 'rb') as f:
            training_data = pickle.load(f)

    with open(args.config, 'r') as f:
        config = json.load(f)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    resFile = args.result
    if not os.path.isfile(resFile):
        results = pd.DataFrame(columns=["Location", "Num Topics", "POS", "Use Translated", "Seed", "Coherence Score", "Mean Score"])
    else:
        results = pd.read_csv(resFile, index_col=0)

    topics = config["Topics"]
    pos_select = config["POS"]
    useTranslated = config["UseTranslated"]==1

    LDA = LDA_Model(training_data=training_data, num_topics=topics, pos=pos_select, translated = useTranslated,
                        datadir = datadir, use_multicore=multi_core, results = results,
                        core=num_core, iterations=iters, save_corpus = save_corpus)
    LDA.save_results(resFile, outdir)