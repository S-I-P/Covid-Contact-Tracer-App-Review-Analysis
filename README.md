# Empirical Study of COVID-19 Contact Tracing App Reviews
A large-scale empirical study of reviews of COVID-19 contact tracing applications by LDA Topic Modeling and Sentiment Analysis.

## Requirements
`google-play-scraper
openpyxl
googletrans
demoji
nltk
pandas
gensim
transformers
scikit-learn`\
To install:
`pip install -r requirements.txt`

## Dataset

Download Google Play Store app reviews through [google-play-scraper](https://github.com/JoMingyu/google-play-scraper).\
Run [scrape.py](scraper/scrape.py):
```
python scrape.py --appids appids.txt --out covid-tracer-apps
```
Here, the first argument is a file containing the appids separated by newline. The scraper code can be paused by keyboard interruption. There will be some files (continuation token and a text file tracking which app to start scraping from) to help the scraper resume later. The scraper saves the reviews in separate folders for each app, and each folder has Excel files that contain a maximum of 200 reviews. After scraping all the reviews, run [combineScraped.py](scraper/combineScraped.py)
```
python combineScraped.py <data-directory> <output-directory>
```
The downloaded raw data used for our experiments: [Raw Data](https://drive.google.com/drive/folders/1gD83lp0n2rlusD-_lJZ_rorSaPuzaeev?usp=share_link)

## Preprocessing
* Discard URLs and emoticons
* Translate using [googletrans](https://github.com/ssut/py-googletrans)
* Remove/modify app-specific numbers and terms ([AppCountry.json](preprocess/AppCountry.json))
* Remove stopwords\
Run:
```
python preprocessLDA.py --data <data-directory> --out <output-directory> --appids <appids-text-file> \
--stat <status-file> --appInfo AppCountry.json --wordmap wordmap.json
```
Here, the input data directory contains the Excel files, which are the output from [combineScraped.py](scraper/combineScraped.py). The preprocessing code can be stopped by keyboard interruption. Running the code later resumes where it was stopped. The status file keeps track of it. [AppCountry.json](preprocess/AppCountry.json) includes specific terms and numbers for each app, while [wordmap.json](preprocess/wordmap.json) includes words and phrases frequent in COVID tracer apps.

Afterwards, get parts of speech of the preprocessed reviews by running [tagPOS.py](preprocess/tagPOS.py):
```
python tagPOS.py <preprocessed-data-directory>
```
Preprocessed data for LDA topic modeling: [Preprocessed LDA](https://drive.google.com/drive/folders/1i5YIAcuG8F4wRKvcI9yOP4k4c4VKPDCV?usp=share_link)

For sentiment analysis preprocessing run:
```
python preprocessSentiment.py --data <preprocessedLDA-data-file> --out <output-directory> \
--appids <appids-text-file> --stat <status-file> --appInfo AppCountry.json --wordmap wordmap.json
```
Here, the data argument refers to the preprocessed data mentioned above.

## LDA Topic Modeling
* Train [gensim](https://radimrehurek.com/gensim/) LDA model
* Change configuration (number of topics, parts of speech (POS), and use_translated) for LDA model training in [config.json](LDA/config.json). Number of topic options; "Topics" is an array of numbers. POS can be an empty array for all POS or an array of selected POS from the 12 possible POS in the NLTK universal tagset, e.g., ["NOUN", "VERB", "ADJ"].

Run:
```
python LDA.py --data <data-directory> --out <output-directory> --result <result-csv-file> \
--config <config-json-file> --multicore <True/False> --core <num-of-cpu-threads> \
--iteration <num-of-iterations> --corpus <corpus-file> --save_corpus <True/False>
```
* Setting the save_corpus option true is helpful when run multiple times.
* For topic inference, run:

```
python assignTopics.py --model <model_file> --out <output-directory> --data <data-directory> \
--seed <seed> --base_topics <topic_labels> --hierarchy <topic-hierarchy> --config <config-file> \
--multicore <True/False> --iteration <number_iterations> --core <num_of_cores>
```

* The model parameter is for saved LDA model, and seed parameter is to create the model with a particular seed.
* The base_topics file contains the n number of topic labels, and the hierarchy file is for assigning each topic to further sub-categories and categories.
* [Labelled topics](https://drive.google.com/drive/folders/14TlxuVA9t-3aFIDiwibTsA4hZkdo4wg2?usp=share_link)
* The [models](https://drive.google.com/drive/folders/1tHuXF7HvfYG22sO_wwsBG4N0hSxLJS7J?usp=drive_link) used for the above labelling

## Sentiment Analysis
We annotated 7060 reviews for sentiment classification. We have used [train](SentimentAnalysis/train_data.xlsx) and [test](SentimentAnalysis/test_data.xlsx) set of sizes 5400 and 1660, respectively.

### Pretrained Transformer Model
Fine-tune a pretrained [transformer](https://huggingface.co/docs/transformers/index) model on the training dataset and evaluate.\
Run:
```
python sentimentClassifyBERT.py --model <0/1> --train <training_data> --test <test_data> --load <load_model> \
--ckpt <checkpoint_save_directory> --output <output_directory> --batch <batch_size> --epoch <epoch>
```
Arguments:
1. model: Which model to train? 0:Bert, 1:Roberta
1. train, test: train and test data files respectively
1. load: start training from this model.
1. ckpt: the intermediate models will save here.

### sklearn classifier
To train the sentiment classifier using [scikit-learn](https://scikit-learn.org) cross-validation of Gradient Boosting Classifier run:
```
python sentimentClassifyGBC-CV.py --data <train-data> --output <output-directory> --search <1/2> \
--verbose <VERBOSE> --threads <Num-CPU-Threads> --iter <Num-Iterations>
```
1. The search argument is for setting the hyper-parameter tuning: 1->Randomized Search or 2->Grid Search. The parameters can be modified in the relevant function. For random search, the iter argument sets the number of iterations.
2. The verbose argument controls the verbosity:\
    \>1: shows computation time and parameter candidates\
    \>2: shows computation time, parameter candidates, and score\
    \>3: the fold and candidate parameter indexes are also displayed together with the starting time of the computation
3. The output directory contains the CV results as a binary object and a csv file, and the best parameters in a txt file.
