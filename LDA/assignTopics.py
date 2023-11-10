from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel as LMSingle
from gensim import corpora

import pickle
from operator import itemgetter
import os
import pandas as pd
import json
import argparse

from nltk.stem.snowball import SnowballStemmer
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

class LDADocument:
    def __init__(self, app, id, body):
        self.app = app
        self.id = id
        self.body = body

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'LDADocument':
            #from settings import LDADocument
            return LDADocument
        return super().find_class(module, name)

def read_data(datadir, useTranslated, pos):
    print(datadir, '----', len(os.listdir(datadir)), 'apps----')
    model_data = []
    dfs = {}

    for file in os.listdir(datadir):
        print(file[:-5])
        df = pd.read_excel(os.path.join(datadir, file))
        dfs[file[:-5]] = df
        nrows = df.shape[0]
        for i in range(nrows):
            if (not useTranslated) and df.loc[i]["isEnglish"]==0:
                continue
            text = df.loc[i]["preprocessedLDA"]
            if not isinstance(text, str) or len(text)==0:
                continue
            review = filter_pos(text.split(), df.loc[i]["POS1"].split(), df.loc[i]["POS2"].split(), pos)
            if len(review)>0:                
                doc = LDADocument(file[:-5], df.loc[i]['reviewId'], review)
                model_data.append(doc)

    return model_data, dfs

def build_corpus(model_data):
    #document_ids = []
    token_collection = []
    for document in model_data:
        #document_ids.append(document.id)
        token_collection.append(stem_tokens(document.body))

    dictionary = corpora.Dictionary(token_collection)
    dictionary.filter_extremes(no_below=20, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in token_collection]

    return corpus, dictionary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA Model')

    parser.add_argument('--model', type=str,
                        help='Saved model file', default=None)

    parser.add_argument('--out', type=str,
                        help='Output Directory', default="Labelled_Topics")

    #parser.add_argument('--corpus', type=str,
    #                    help='Saved corpus file', default=None)

    parser.add_argument('--data', type=str,
                        help='Data Directory', default="Data/Europe")

    parser.add_argument('--seed', type=int,
                        help='Seed', default=None)

    parser.add_argument('--base_topics', type=str,
                    help='Base topics', default="base_topics/Europe.txt")

    parser.add_argument('--hierarchy', type=str,
                    help='Topic hierarchy', default="hierarchy.json")

    parser.add_argument('--config', type=str,
                    help='Configuration File', default="config.json")

    parser.add_argument('--multicore', type=bool,
                        help='Multicore', default=True)

    parser.add_argument('--iteration', type=int,
                        help='Number of iterations', default=100)

    parser.add_argument('--core', type=int,
                        help='CPU Threads', default=8)

    args = parser.parse_args()

    if (args.model is None) and (args.seed is None):
        print("No model file or seed")
        exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)
    useTranslated = config["UseTranslated"]==1
    pos_select = config["POS"]

    #if args.corpus is not None:
    #    with open(args.corpus, 'rb') as f:
    #        data = pickle.load(f)
    #else:
    data, dfs = read_data(args.data, useTranslated, pos_select)

    with open(args.base_topics, 'r') as f:
        labels = [topic for topic in f.read().splitlines()]
    topics_count = len(labels)

    with open(args.hierarchy, 'r') as f:
        hierarchy = json.load(f)
    irrelevant_topics = set(labels).difference(set(hierarchy.keys()))
    for irr in irrelevant_topics:
        hierarchy[irr] = {"Sub": "Irrelevant", "Cat": "Irrelevant"}

    if args.model is not None:
        if args.multicore:
            model = LdaMulticore.load(args.model)
        else:
            model = LMSingle.load(args.model)
        corpus = [model.id2word.doc2bow(text.body) for text in data]
    else:
        corpus, dictionary = build_corpus(data)
        if args.multicore:
            model = LdaMulticore(corpus, num_topics=topics_count, id2word=dictionary,
                                passes=50, workers=args.core, alpha='symmetric',
                                random_state=args.seed, eta='auto', iterations=args.iteration)
        else:
            model = LMSingle(corpus =corpus, num_topics=topics_count, id2word=dictionary,
                                random_state=args.seed, passes=50, alpha='auto',
                                eta='auto', iterations=args.iteration)

    numTopics = len(model.print_topics(-1))
    print("Number of topics=", numTopics)
    all_topics = model.get_document_topics(corpus, per_word_topics=True)
    print('Total reviews:', len(all_topics))
    print('---------------')
    print(labels)

    outdir = args.out
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    dfout = pd.DataFrame(columns=["appId", "reviewId", "userName", "content", "score", "thumbsUpCount",
                                    "time", "reviewCreatedVersion", "preprocessedLDA", "translation",
                                    "topicId", "topicLabel", "probability", "subCategory", "category"])
    outIdx = 0

    for i in range(len(all_topics)):
        tid_ = []
        tprb_=[]
        tlabel_ = []
        sc_=[]
        cat_=[]
        doc_topics = all_topics[i][0]
        doc_topics = sorted(doc_topics,key=itemgetter(1),reverse=True)
        maxProb = doc_topics[0][1]
        for dt in doc_topics:
            currProb = dt[1]
            if currProb!=maxProb:
                break
            tid_.append(str(dt[0]))
            tprb_.append(str(dt[1]))
            tlabel_.append(labels[dt[0]])
            sc_.append(hierarchy[labels[dt[0]]]['Sub'])
            cat_.append(hierarchy[labels[dt[0]]]['Cat'])
        
        topic_id = '_'.join(tid_)
        topic_label = '_'.join(list(set(tlabel_)))
        topic_prob = '_'.join(tprb_)
        topic_sub = '_'.join(list(set(sc_)))
        topic_cat = '_'.join(list(set(cat_)))
        appid = data[i].app
        reviewId = data[i].id
        rowidx = dfs[appid].index[dfs[appid]['reviewId']==reviewId][0]
        
        dfout.loc[outIdx] = [appid, reviewId] + dfs[appid].loc[rowidx].values.flatten().tolist()[1:9] + [topic_id, topic_label, topic_prob, topic_sub, topic_cat]
        outIdx+=1
    dfout.to_excel(outdir+'/'+ args.data[args.data.rfind('/')+1:] +'.xlsx', index=False)