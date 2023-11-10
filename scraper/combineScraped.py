import pandas as pd
import os
import sys

if __name__ == '__main__':
    #directory containing previously scraped reviews files
    if len(sys.argv)<2:
        print("Data directory not given")
        exit(1)
    dataDir = sys.argv[1]
    #output directory; optional
    outDir = 'scraped-apps'
    if len(sys.argv)>2:
        outDir = sys.argv[2]
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    #combine review files into one
    for appid in os.listdir(dataDir):
        print(appid)
        fpath = os.path.join(dataDir, appid)
        dfs = []
        for file in os.listdir(fpath):
            df = pd.read_excel(os.path.join(fpath, file))
            dfs.append(df)

        allReviews = pd.concat(dfs)
        allReviews.to_excel(os.path.join(outDir, appid+'.xlsx'), index=False)