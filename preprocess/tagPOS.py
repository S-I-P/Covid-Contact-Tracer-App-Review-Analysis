from nltk import pos_tag
import os
import sys
import pandas as pd

if len(sys.argv)<2:
    print("Usage: python tagPOS.py <data-directory>")
    exit(1)

files = os.listdir(sys.argv[1])

for file in files:
    f = os.path.join(sys.argv[1], file)
    df = pd.read_excel(f)
    print(file, df.shape)
    print('--------------')
    #df["POS1"] = ""
    #df["POS2"] = ""
    for i in range(df.shape[0]):
        text = df.loc[i]["preprocessedLDA"]
        if not isinstance(text, str) or len(text)==0:# or df.loc[i]["isEnglish"]==0:
            continue

        text = text.encode('ASCII', 'ignore').decode().strip()
        tokens = text.split()
        tags1 = pos_tag(tokens, tagset= 'universal')
        tags1 = [t[1] for t in tags1]
        tags2 = []
        for t in tokens:
            tags2.append(pos_tag([t], tagset= 'universal')[0][1])
        
        df.loc[i, "preprocessedLDA"] = text
        df.loc[i, "POS1"] = ' '.join(tags1)
        df.loc[i, "POS2"] = ' '.join(tags2)
    df.to_excel(f, index=False)