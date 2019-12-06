import pandas as pd
import acquire

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# basic_clean takes in a string and apply some basic text cleaning to it
def basic_clean(string):
    '''
    Put doctring comments here. That way I can reference it for later to explain my function
    '''
    
    # lowercase capitalized letters
    string = string.lower()
    
    # normalizing the data by removing non-ASCII characters
    string = unicodedata.normalize('NFKD', string)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')

    # remove whitespace
    string = string.strip()

    # remove anything that is not a through z, a number, a single quote, or whitespace
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    
    # convert newlins and tabs to a single space
    string = re.sub(r'[\r|\n|\r\n]+',' ', string)
    
    return string

# tokenize takes in a string and tokenize all the words in the string
def tokenize():
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)

# stem accepts some text and return the text after applying stemming to all the words.
def stem():
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string_of_stems = ' '.join(stems)
    return string_of_stems

# lemmatize accepts some text and return the text after applying lemmatization to each word
def lemmatize():
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string_of_lemmas = ' '.join(lemmas)
    return string_of_lemmas

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    # Tokenize the string
    string = tokenize(string)

    words = string.split()
    stopword_list = stopwords.words('english')

    # remove the excluded words from the stopword list
    stopword_list = set(stopword_list) - set(exclude_words)

    # add in the user specified extra words
    stopword_list = stopword_list.union(set(extra_words))

    filtered_words = [w for w in words if w not in stopword_list]
    final_string = " ".join(filtered_words)
    return final_string

def prep_articles(df):
    df["original"] = df.body
    df["stemmed"] = df.body.apply(basic_clean).apply(stem)
    df["lemmatized"] = df.body.apply(basic_clean).apply(lemmatize)
    df["clean"] = df.body.apply(basic_clean).apply(remove_stopwords)
    df.drop(columns=["body"], inplace=True)
    return df

def prep_blog_posts():
    df = acquire.get_blog_posts()
    return prep_articles(df)

def prep_news_articles():
    df = acquire.get_news_articles()
    return prep_articles(df)

def prep_corpus():
    blog_df = prep_blog_posts()
    blog_df["source"] = "Codeup Blog"

    news_df = prep_news_articles()
    news_df["source"] = "InShorts News"

    return blog_df, news_df