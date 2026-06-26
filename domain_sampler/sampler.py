import pandas as pd
import re
import random
import torch
import numpy as np
from trafilatura import extract
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import seaborn as sns
from sentence_transformers import SentenceTransformer
#from datetime import datetime
from scipy import stats
import math
from bs4 import BeautifulSoup
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
tqdm.pandas()


class DomainSampler():
    # getting the stop words to be used later
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    global_stop_words_set = set()
    for lang in stopwords.fileids():
        global_stop_words_set.update(stopwords.words(lang))

    def __init__(self, bert_model_path:str = "safe_bertopic", confidence:float = 0.95, margin_error:float = 0.05,
                  embedding_model = None, embeddings:list[float] = []):
        ''' Initializes the DomainSampler with population size, confidence level, margin of error, embedding model, and precomputed embeddings.
        :param bert_model_path: Path to the BERTopic model (optional)
        :param confidence: Confidence level (e.g., 0.95 for 95%) (optional)
        :param margin_error: Acceptable margin of error (e.g., 0.05 for 5%) (optional)
        :param embedding_model: The embedding model to use (optional)
        :param embeddings: A list of precomputed embeddings to use for topic modeling (optional)'''
        self.confidence = confidence
        self.margin_error = margin_error
        self.bert_model_path=bert_model_path
        if not embeddings:
            self.embeddings = None #this value forces the modelr to use the default embedding model to generate the embeddings for topic modeling
        else:
            self.embeddings = np.array(embeddings)
        # downloading the embedding model
        self.embedding_model = embedding_model
        if not self.embedding_model:
            if torch.cuda.is_available():
                self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                                                    device="cuda:0")
            else:
                self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        #loading the topic modeler
        try:
            self.topic_model = BERTopic.load(self.bert_model_path, embedding_model = self.embedding_model)
            self.topic_representations = self.topic_model.get_topic_info() #shows the representative words per topic
        except:
            raise ValueError(f"BERTopic model could not be loaded from the path: {self.bert_model_path}. Please check the path and try again.")
    @staticmethod
    def get_min_sample_size(pop_size: int, confidence: float, margin_error: float)->int:
        """
        Calculates minimum sample size for a finite population.

        :param pop_size: Total number of individuals in the population
        :param confidence: Confidence level (e.g., 0.95 for 95%)
        :param margin_error: Acceptable margin of error (e.g., 0.05 for 5%)
        """
        # 1. Get Z-score based on confidence level
        # Use 1 - (1 - confidence) / 2 to get the two-tailed critical value
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        # 2. Set estimated proportion (0.5 provides the safest/maximum sample size)
        p = 0.5

        # 3. Cochran's formula for infinite population
        n_0 = (z**2 * p * (1 - p)) / (margin_error**2)

        # 4. Adjust for finite population (Finite Population Correction)
        n = n_0 / (1 + ((n_0 - 1) / pop_size))

        return math.ceil(n)
    
    def clean_text(self, text: str, stop_words = global_stop_words_set):
        '''
        remove stop words, urls and numbers to prepare test for topic modeling

        :param text: takes tha extracted article text from HTML
        :param stop_words: takes the stop words from all languages
        '''
        if not isinstance(text, str):
            # raise TypeError("text must be a string")
            return ""

        # remove urls
        text = re.sub(r"http\S+", " link ", text)

        # replace any digit with 'number'
        text = re.sub(r"\d+", " number ", text)

        # set space before and after any punctuation
        text = re.sub(r"([^\w\s])", r" \1 ", text)

        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        text = " ".join([ w for w in filtered_tokens if len(w) > 1 ] )
        return text.lower().strip()
    
    def preprocess_html(self, articles:list)->list[dict]:
        """
        Preprocesses the HTML content of articles from a domain and returns all the info and cleaned text.
        """        
        articles_text = []
        for response_html in tqdm(articles):
            try:
                result = json.loads(extract(response_html, with_metadata=True, output_format="json", include_comments=False,
                                            include_tables=False))
                content = self.clean_text(result["text"])
                articles_text.append({"title":result["title"],
                        "author":result["author"],
                        "domain_name":result["hostname"],
                        "date":result["date"],
                        "categories": result["categories"],
                        "tags":result["tags"],
                        "text":str(result["title"])+' '+str(result["categories"])+' ' +result["tags"]+' '+content})
            except:
                    soup = BeautifulSoup(response_html, "html.parser")
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    articles_text.append({"title":None,
                        "author":None,
                        "domain_name":None,
                        "date":None,
                        "categories": None,
                        "tags":None,
                        "text":self.clean_text(soup.get_text(separator=" ", strip=True))})
        return articles_text
    
    def topic_analysis(self, urls:list[str], articles:list,embedings:list=None,deep_analysis:bool=False,is_html=True)-> list[pd.DataFrame]:
        """
        Preprocesses the HTML content of articles from a domain and returns topic with their relative urls ordered by 
        the number of words of their corresponding article content.

        :param urls: list fo the urls sample from limited population theory
        :param articles: list of articles HTML content
        :param deep_analysis: flag to indicate whether to perform deep analysis 
        and get top 3 topic with their probabilities. However, you can't use pre-calculated embeddings with it.
        """ 
        # Validate that urls and articles have the same length
        if len(urls) != len(articles):
            raise ValueError(f"The number of URLs ({len(urls)}) must match the number of articles ({len(articles)})")
        if is_html:
            cleaned_articles = [art['text'] for art in tqdm(self.preprocess_html(articles))]
        else:
            cleaned_articles=articles

        art_lens = [len(art.split()) for art in cleaned_articles]

        if deep_analysis:
            topic_distr, _ = self.topic_model.approximate_distribution(cleaned_articles,
                                                        window=4, stride=1, use_embedding_model=True)
            top_3_topics, top_3_probabs = [], []
            for i, doc_distribution in enumerate(topic_distr):
                top_3_indices = np.argsort(doc_distribution)[::-1][:3]
                output_probabilities = [float(doc_distribution[topic_id]) for topic_id in top_3_indices]
                top_3_topics.append(top_3_indices.tolist())
                top_3_probabs.append(output_probabilities)
            topics = [topics[0] for topics in top_3_topics]
            probs = [probs[0] for probs in top_3_probabs]
            analysis_df = pd.DataFrame({'url':urls, 'article_word_num':art_lens, 'topic' : topics, 'prob': probs,
                                            'top_3_topics': top_3_topics, 'top_3_probabs': top_3_probabs})
        else:
            if embedings:
                # self.embeddings = np.array(embedings)
                self.embeddings = np.array([elem[0:384] for elem in  embedings])
            topics, probs = self.topic_model.transform(cleaned_articles, embeddings = self.embeddings)
            analysis_df = pd.DataFrame({'url':urls, 'article_word_num':art_lens, 'topic' : topics, 'prob': probs})
        # 1. Sort the dataframe by the number of words in ascending order
        analysis_df = analysis_df.sort_values(by="article_word_num", ascending=True)
        # 2. Group by topic and aggregate the URLs into a list
        result_df = (
            analysis_df.groupby("topic")["url"]
            .apply(list)
            .reset_index(name="ordered_urls"))        
        return [analysis_df, result_df]
    def topic_analysis_cosin_sim(self,emb_lst:list,topk=3):
        # Get topic info
        if len(emb_lst)>0:
            topic_info = self.topic_model.get_topic_info()
            topic_names = dict(zip(topic_info['Topic'], topic_info['Name']))
            embeddings = np.array([elem[0:384] for elem in  emb_lst])
            # Get all topic probabilities using cosine similarity
            topic_embeddings = self.topic_model.topic_embeddings_
            similarities = cosine_similarity(embeddings, topic_embeddings)
            exp_similarities = np.exp(similarities)
            all_topic_probs = exp_similarities / exp_similarities.sum(axis=1, keepdims=True)

            # print(f"Full distribution shape: {all_topic_probs.shape}")
            # print(f"All probabilities:\n{all_topic_probs}")
            # Get top 3 topics per document
            results = []
            for doc_idx, doc_probs in enumerate(all_topic_probs):
                top_k_indices = np.argsort(doc_probs)[-topk:][::-1] 
                top_k_probs= np.sort(doc_probs)[-topk:][::-1]
                doc_topics=[]
                # doc_topics.append(doc_idx)            
                for rank, topic_id in enumerate(top_k_indices, 1):
                    prob = doc_probs[topic_id]                
                    name = topic_names.get(topic_id, "Unknown")                
                    doc_topics.extend([topic_id,name,top_k_probs[rank-1]])
                results.append(doc_topics)   
            
            titles=[]          
            for i in range(0,topk):
                titles.extend([f"topic{i}_id",f"topic{i}_name",f"topic{i}_prop"]) 
            df = pd.DataFrame(results,columns=titles)
            return df
        return None