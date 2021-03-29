# Run in python console
import nltk; nltk.download('stopwords')
from numba import cuda
from numba import jit
import ru2
import re
import numpy as np
import pandas as pd
from pprint import pprint 
import gensim
import gensim.corpora as corpora
import scipy
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LdaMulticore
# spacy for lemmatization
import spacy
import json
spacy.prefer_gpu()

#nlp = spacy.load("en_core_web_sm")

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import xlrd
import xlwt


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) 

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')

data=[]
stopwordsdata=[]

wb = xlrd.open_workbook('xl.xls')
sheet = wb.sheet_by_index(0)
for rownum in range(sheet.nrows):
    row = sheet.row_values(rownum)
    for c_el in row:
        data.append(c_el)
 
sheet = wb.sheet_by_index(1)
for rownum in range(sheet.nrows):
    row = sheet.row_values(rownum)
    for c_el in row:
        stopwordsdata.append(c_el)

stop_words.extend(stopwordsdata)
#df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
#print(df.target_names.unique())
#df.head()
 
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Define functions for stopwords, bigrams, trigrams and lemmatization

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
#-----------------------------

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
         
data_words = list(sent_to_words(data))
print("DataWordz created")
data_words_nostops = remove_stopwords(data_words)# Remove Stop Words

# ----Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words_nostops], threshold=100)  
# ----Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
 

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
data_words_trigrams = make_trigrams (data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
print("Bi/tri-grams created")

nlp = spacy.load('ru2')
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized = remove_stopwords(data_lemmatized)# Remove Stop Words
print("Data's lemmatization ended")

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
id2word.filter_extremes( no_below=20, no_above=0.5) 

id2word.save("id2word.dict")
print("Dictionary created and saved")
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
corpus=gensim.corpora.MmCorpus.serialize("data.mm", corpus)
#print(corpus[:1])
corpus=gensim.corpora.MmCorpus("data.mm")
print("corpus created and saved")

try:
    lda_model=LdaModel.load("lda_model.model")
    print("LDA...'s loaded!")
except:

    print("LDA...shall be created...")
    # Human readable format of corpus (term-frequency)
    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
    print("start LDA... ")
    # Build LDA model
    num_topics = 40
    chunksize = 820 
    passes = 20
    iterations = 500
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    lda_model =  LdaModel(
                            corpus=corpus,
                            id2word=id2word,
                            chunksize=chunksize,
                            alpha='auto',
                            eta='auto',
                            iterations=iterations,
                            num_topics=num_topics,
                            passes=passes,
                            eval_every=eval_every
                        )
    lda_model.save('lda_model.model')
    print("LDA...'s been created!")
    print("LDA...'s saved!")
    
    #ldamodel.LdaModel
    #filename = os.path.join(output_dir, ) % (model.K, model.M))
    #lda_model.save_result("Kd_Md.json")

print("stop LDA... ")
top_topics = lda_model.top_topics(corpus)
pprint(top_topics)
 
lda_model.print_topics()
pprint(lda_model.print_topics())

#Отображение доминирующей темы
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)
# Format    
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
 
df=pd.DataFrame(df_dominant_topic)
df
df.to_excel("df_dominant_topic+.xls")
print("Dominant topic saved!")    
 



def compute_coherence_values(dictionary, corpus, texts, limit, start=20, step=2):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    sent_topics_df = pd.DataFrame()
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(
                            corpus=corpus,
                            id2word=id2word,
                            alpha='auto',
                            eta='auto',
                            num_topics=num_topics,

                        )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=20, limit=40, step=2)
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# Show graph
limit=40; start=20; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()