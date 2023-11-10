from google_play_scraper import reviews
import pandas as pd
import pickle
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='app review scraper')
    parser.add_argument('--appids', type=str,
                        help='appid text file', default="appids.txt")

    parser.add_argument('--out', type=str,
                        help='Output Directory', default="covid-tracer-apps")
    args = parser.parse_args()
    #read appids from textfile
    with open(args.appids, 'r') as f:
        appids = f.read().splitlines()
    #output directory
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    #find which app to start scraping from
    #if scraping was interrupted before last.txt has the name of the last app with scraping completed
    #default start from beginning
    start = 0
    if not os.path.isfile('last.txt'):
        currfile = open('last.txt', 'w')
    with open('last.txt', 'r') as f:
        startFrom = f.readlines()
    if len(startFrom)>0:
        start= appids.index(startFrom[0])
    appids = appids[start:]
    #google-play-scraper fetches 200 reviews each time
    #continuation token helps to keep track
    #also helpful if scraping is interrupted
    count = 200
    continuation_token = None
    #read continuation_token
    if os.path.isfile('continuation_token'):
        with open('continuation_token', 'rb') as tokenfile:
            continuation_token = pickle.load(tokenfile)

    for appid in appids:
        print(appid)
        dfIdx = 0
        #start from this app in case of stopping scraper
        with open('last.txt', 'w') as f:
            f.write(appid)
        #out file path
        savePath = os.path.join(args.out, appid)
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        else:
            dfIdx = len(os.listdir(savePath))
        #scrape 200 reviews each time and save those
        length = count
        while length==count:
            df = pd.DataFrame(columns=["reviewId", "userName", "content","score", "thumbsUpCount", "time", "reviewCreatedVersion"])
            result, continuation_token = reviews(
                appid,
                count=count,
                continuation_token = continuation_token
            )
            
            for i, res in enumerate(result):
                df.loc[i] = [res["reviewId"], res["userName"], res["content"], res["score"], res["thumbsUpCount"], res["at"], res["reviewCreatedVersion"]]

            length=len(result)
            dfIdx += 1
            print('scraped ', length)
            df.to_excel(savePath+'/'+str(dfIdx)+'.xlsx', index=False)
            #save continuation_token
            with open('continuation_token', 'wb') as tokenfile:
                pickle.dump(continuation_token, tokenfile)
        #curr appid scraping done
        continuation_token = None