import os
import json
import argparse
import pandas as pd
import re

from nltk import WordPunctTokenizer
punct_tokenizer = WordPunctTokenizer()

stopwords =['me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            "'s", 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'had',
            'do', 'doing', 'an', 'the', 'and', 'if', 'or', 'because',
            'as', 'while', 'of', 'at', 'by', 'with', 'about', 'against', 'between',
            'into', 'through', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'over', 'under', 'further', 'then', 'here', 'there',
            'when', 'where', 'why', 'how', 'both', 'each', 'more', 'other',
            'such', 'own', 'so', 'some',
            'can', 'will', 'just', 'should', 'now', 'll', 're', 've', 'could', 'ma', 
            'would', 'omg', 'idk', 'app']

url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
rep_regex = re.compile(r"([A-Za-z])\1{2,}", re.DOTALL)
phone_regex1 = re.compile('((samsung)|(iphone)|(ios)|(oneplus)|(huawei)|(redmi)|(oppo)|(sony)|(htc))( )*(.*\d)[a-z]* ')
phone_regex2 = re.compile('(samsung)|(iphone)|(ios)|(oneplus)|(huawei)|(redmi)|(oppo)|(sony)|(htc)')
phone_regex3 = re.compile('(phone( )*)+')

def replaceRepeat(s):
    return rep_regex.sub(r"\1\1", s)

def remove_url(s):
    return url_regex.sub(" ",s)

def remove_app_stopwords(s, stops):
    s1 = str(s)
    for stop in stops:
        s1 = s1.replace(" "+stop+" ", " ")
        if s1.startswith(stop+" "):
            s1 = s1.replace(stop+" ", "", 1)
    return s1

def remove_device_names(s):
    s = phone_regex1.sub("phone ", s)
    s = phone_regex2.sub("phone ", s)
    return phone_regex3.sub("phone ", s)

def tokenize(text):
    #toks = word_tokenize(text)
    toks = text.split()
    tokens = []
    for tok in toks:
        if tok in wm_keys:
            tokens += wordmap[tok].split()
        else:
            tokens += punct_tokenizer.tokenize(tok)
    
    tokens = [tok for tok in tokens if tok.isalpha() or '_' in tok]  # lower case, remove number, punctuation
    return tokens

def remove_stopwords(tokens):
    stopped_tokens = [tok for tok in tokens if not tok in stopwords]
    return stopped_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Sentiment Analysis')
    parser.add_argument('--data', type=str,
                        help='Preprocessed(LDA) Data Directory', default="Preprocessed")
    parser.add_argument('--out', type=str,
                        help='Output Directory', default="Preprocessed_SentimentAnalysis")
    parser.add_argument('--appids', type=str,
                        help='appid text file', default="appids.txt")
    parser.add_argument('--stat', type=str,
                        help='Completion status file', default="complete.txt")
    parser.add_argument('--appInfo', type=str,
                        help='App info stopwords and numbers file', default="AppCountry.json")
    parser.add_argument('--wordmap', type=str,
                        help='Word map file', default="wordmap.json")

    args = parser.parse_args()

    datadir = args.data
    outdir = args.out
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    with open(args.appids, 'r') as f:
        appIds = f.read().splitlines()
    stat = args.stat
    lastApp = appIds[0]
    start = 0
    if os.path.isfile(stat):
        with open(stat, 'r') as f:
            done = f.read().split()
        lastApp = done[0]
        start = int(done[1])

    with open(args.appInfo, 'r') as f:
        appInfo = json.load(f)

    global wordmap
    global wm_keys
    wordmap = {}
    with open(args.wordmap, 'r') as f:
        wordmap = json.load(f)

    wm_keys = list(wordmap.keys())

    appIds = appIds[appIds.index(lastApp):]
    print(len(appIds), 'Left\n-----------------------')
    print(appIds)
    print('-------------------------')
    
    #append one character word to stopwords
    for ch in range(26):
        stopwords.append(chr(ord('a')+ch))

    for itr, appId in enumerate(appIds):
        try:
            print('-----------------------------------\n', itr, appId, '\n--------------------------------\n')
            file = appId + '.xlsx'
            dest = os.path.join(outdir, file)
            df = pd.read_excel(os.path.join(datadir, file))
            nrows = df.shape[0]
            print('Total:', nrows)
            print('Start:', start)
            if start==0:
                #new
                dfout = df.copy()
                #initialize preprocessed to empty
                dfout = dfout.drop(columns=['preprocessedLDA', 'POS1', 'POS2'])
                dfout["preprocessedSA"] = ""
            else:
                #resume
                dfout = pd.read_excel(dest)

            info = appInfo[appId]
            for i in range(start, nrows):
                comment = dfout.loc[i]["translation"]
                if not isinstance(comment, str) or len(comment)==0:# or df.loc[i]["isEnglish"]==0:
                    continue
                #replace letters occuring more than twice continuously to only two letters
                comment = replaceRepeat(comment)
                #remove app related stopwords
                comment = remove_app_stopwords(comment, info['stopwords'])
                #replace device model with phone
                comment = remove_device_names(comment)
                #tokenize
                comment = tokenize(comment)
                #stopwords
                comment = remove_stopwords(comment)
                dfout.loc[i,"preprocessedSA"] = ' '.join(comment)                                
            
            dfout.to_excel(dest, index=False)
            start = 0

        except KeyboardInterrupt:
            print(appId, i)
            with open(stat, 'w') as f:
                f.write(appId+' '+str(i))
            dfout.to_excel(dest, index=False)
            break