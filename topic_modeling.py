import findspark
findspark.init()
findspark.find()

from pyspark.sql.types import StructType,StructField, StringType, ArrayType, IntegerType, FloatType
from pyspark.sql import SparkSession, Row

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('WARN')
PLOT_FOLDER = 'plots/'


'''
Data Loading
'''
def load_data(path):
    def printRdd(rdd):
        for r in rdd:
            print(r)
            print()

    def convertTextReviewToList(textReview):
            lines = textReview.split('\n')
            valList = list()
            try:
                    memory = ''
                    for l in lines:
                            keyValue = l.split(':', 1)
                            # assert(len(keyValue) == 2)
                            if (len(keyValue) == 2):
                                key = keyValue[0].split('/', 1)[1]
                                value = memory + keyValue[1].strip()
                                memory = ''
                                # obj[key] = value
                                valList.append(value)
                            else:
                                memory = memory + keyValue[0]
                    
                    if (len(valList) != 8):
                        valList = list(textReview)
            except:
                    # obj["failedToParse"] = textReview
                    # valList.append(textReview)
                    valList = list(textReview)
            return valList
        
    spark.sparkContext._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n")
    dataRdd = spark.sparkContext.textFile(path)
    dataRdd = dataRdd.map(lambda x: (convertTextReviewToList(x)))

    # filter out reviews that can't be parsed
    dataRdd = dataRdd.filter(lambda y: len(y)==8)

    # set up dataframe
    reviewSchema = StructType([
        StructField('ProductId', StringType(), True),
        StructField('UserId', StringType(), True),
        StructField('ProfileName', StringType(), True),
        StructField('Helpfulness', StringType(), True),
        StructField('Score', StringType(), True),
        StructField('Time', StringType(), True),
        StructField('Summary', StringType(), True),
        StructField('Text', StringType(), True),
    ])

    fullReviewsDf = spark.createDataFrame(dataRdd, schema=reviewSchema) # 'ProductId', 'UserId', 'ProfileName', 'Helpfulness', 'Score', 'Time', 'Summary', 'Text'
    fullReviewsDf = fullReviewsDf.withColumn("Score", fullReviewsDf["Score"].cast(IntegerType()))
    print('Dataframe length:', fullReviewsDf.count())
    return fullReviewsDf

def split_reviews(df):
    '''Split df by positive and negative score'''
    posDf = df.rdd.filter(lambda x: x['Score'] >= 3).toDF()
    negDf = df.rdd.filter(lambda x: x['Score'] < 3).toDF()
    return posDf, negDf



'''
Preprocessing
'''
import re
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
stop_words_english = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess(df):
    '''Preprocess reviews
    remove punctuations, stopwords
    tokenize word'''
    def feature_extraction(s):
        s = re.sub(r'[^\w\s]',' ',s)
        s = re.sub(r'https?:\/\/.*[\r\n]*','',s)
        s = re.sub(r'\d+', '', s)
        a = word_tokenize(s)
        
        # remove words from reviews
        temp_lis = ['br', 'el', 'movie', 'film', 'u', 'p', 'quot', 'le', 'one', 'first', 'really', 'mr', 'john']
        final_lis = []
        for x in a:
            x = x.lower()
            if x not in stop_words_english and x not in temp_lis:
                lem_word = wordnet_lemmatizer.lemmatize(x)
                final_lis.append(lem_word)
        return final_lis
    tokenizedRdd = df.rdd.map(lambda x: Row(ProductId=x['ProductId'], UserId=x['UserId'], Text_FilteredTokens=feature_extraction(x['Text']), Score=x['Score'])) # (product, user, filtered_tokens)

    Schema = StructType([
        StructField("ProductId", StringType(), True),
        StructField("UserId", StringType(), True),
        StructField("Text_FilteredTokens", ArrayType(StringType()), True),
        StructField("Score", IntegerType(), True),
    ])
    tokenizedDf = spark.createDataFrame(tokenizedRdd, schema=Schema)
    return tokenizedDf



'''
LDA
'''
import numpy as np
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import Vectors, SparseVector, DenseVector
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer

def vectorize_df(tokenizedDf):
    '''Convert each review to word count'''
    wordVectorizer = CountVectorizer(inputCol='Text_FilteredTokens', outputCol='Text_Vector', vocabSize=300, minDF=1, minTF=4)
    wordVectorizerModel = wordVectorizer.fit(tokenizedDf)
    # vectorVocabulary = wordVectorizerModel.vocabulary

    vectorizedDf = wordVectorizerModel.transform(tokenizedDf)
    return vectorizedDf, wordVectorizerModel

def lda_fit(vectorizedDf, vectorVocabulary, n_topics):
    '''Fit training data to LDA topic modeling'''
    lda = LDA(featuresCol='Text_Vector',
                        k=n_topics, 
                        maxIter=30,
                        seed=42, 
                        optimizer="online",
                        learningDecay=0.51,
                        subsamplingRate=0.5,
                        optimizeDocConcentration=True)
    ldaModel = lda.fit(vectorizedDf)

    ldaTopicsDf = ldaModel.describeTopics(n_topics)
    # print('Topics identified by LDA')
    # ldaTopicsDf.show(truncate=True) # topic, termIndices, termWeights

    termsOfTopicsRdd = ldaTopicsDf.rdd.map(lambda x: x['termIndices']).map(lambda y: [vectorVocabulary[i] for i in y])
    print('Terms of each topic identified by LDA')
    termsOfTopicsDf = termsOfTopicsRdd.toDF()
    termsOfTopicsDf.show() # topic 1 -> k

    return ldaModel, ldaTopicsDf

def lda_transform(vectorizedDf, ldaModel):
    '''Make prediction on new data'''
    topicScoresDf = ldaModel.transform(vectorizedDf)

    # determining the topics
    def findTopic(v: DenseVector):
        npArr = v.toArray()
        if npArr.sum() == 0:
            return None
        idxMax = np.argmax(npArr, axis=0)
        return idxMax.item()
    topicRdd = topicScoresDf.rdd.map(lambda x: Row(ProductId=x['ProductId'], UserId=x['UserId'], topic=findTopic(x['topicDistribution']), length=len(x['Text_FilteredTokens']), score=x['Score']))
    topicRdd = topicRdd.filter(lambda x: x['topic'] is not None)
    topicDf = topicRdd.toDF()
    return topicDf, topicScoresDf



'''
Visualization Data Transformation
prepare data for visualization
'''
import pandas as pd
from pyspark.sql.functions import explode, monotonically_increasing_id

def count_term_frequency(vectorizedDf, vectorVocabulary):
    '''Count term frequency of Text_FilteredTokens'''
    vocabDf = vectorizedDf.select('Text_FilteredTokens').rdd.flatMap(lambda x: x[0]).toDF(schema=StringType()).toDF('terms')
    vocabFrequency = vocabDf.rdd.countByValue()
    pdf = pd.DataFrame({
                    'term': list(vocabFrequency.keys()),
                    'frequency': list(vocabFrequency.values())
            })
    termFrequencyDf = spark.createDataFrame(pdf).orderBy('frequency', ascending=False)
    def getVocabIndex(term):
        try:
            index = vectorVocabulary.index(term)
            return index
        except:
            return None
    termFrequencyDf = termFrequencyDf.rdd.map(lambda x: Row(term=x['term'][0], frequency=x['frequency'], index=getVocabIndex(x['term'][0]))).toDF()
    return termFrequencyDf

def get_cloud_df(ldaTopicsDf, vectorVocabulary):
    '''Prepare df to plot word cloud'''
    # explode array terms of each topic to row
    explodeTermIndices = ldaTopicsDf.select(ldaTopicsDf.topic, explode(ldaTopicsDf.termIndices)).withColumnRenamed("col","termIndices")
    explodeTerm = explodeTermIndices.rdd.map(lambda x: Row(topic=x['topic'], term=vectorVocabulary[x['termIndices']])) #term=Row(vectorVocabulary[x['termIndices']])
    explodeTerm = explodeTerm.toDF().withColumn("id", monotonically_increasing_id())

    explodeTermWeights = ldaTopicsDf.select(explode(ldaTopicsDf.termWeights)).withColumnRenamed("col","termWeights")
    explodeTermWeights = explodeTermWeights.withColumn("id", monotonically_increasing_id())

    cloudDf = explodeTerm.join(explodeTermWeights, 'id', 'outer').drop('id').orderBy('topic')
    return cloudDf



'''
Visualization
'''
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] * 2 # more colors: 'mcolors.XKCD_COLORS'

def plot_word_count(topicScoresDf, df_type):
    '''Plot word count for each topic'''
    topic_df = topicScoresDf.limit(5000).toPandas()
    review_lens = topic_df.length

    plt.figure(figsize=(12,8))
    plt.hist(review_lens, bins=500, color='navy')
    x_pos = 550
    y_pos = 150
    plt.text(x_pos, y_pos, "Mean   : " + str(round(np.mean(review_lens))))
    plt.text(x_pos, y_pos-10, "Median : " + str(round(np.median(review_lens))))
    plt.text(x_pos, y_pos-20, "Stdev   : " + str(round(np.std(review_lens))))
    plt.text(x_pos, y_pos-30, "1%ile    : " + str(round(np.quantile(review_lens, q=0.01))))
    plt.text(x_pos, y_pos-40, "99%ile  : " + str(round(np.quantile(review_lens, q=0.99))))
    plt.gca().set(xlim=(0, 800), ylabel='Number of Reviews', xlabel='Review Word Count')
    plt.title('Distribution of Review Word Counts', fontsize=16)
    plt.savefig(PLOT_FOLDER + df_type + 'Review Word Count',  dpi=300)
    plt.show()

def plot_word_count_by_topic(topicScoresDf, n_topics, df_type):
    '''plot distribution of word count by LDA topic'''
    topic_df = topicScoresDf.limit(5000).toPandas()
    fig, axes = plt.subplots(2, int(n_topics/2), figsize=(n_topics*1.5,6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        topic_df_sub = topic_df.loc[topic_df.topic == i, :]
        review_lens = topic_df_sub.length
        ax.hist(review_lens, bins=50, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(review_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 300), xlabel='Review Word Count')
        ax.set_ylabel('Number of Reviews', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle('Distribution of Review Word Counts by LDA Topic')
    plt.savefig(PLOT_FOLDER + df_type + 'Distribution of Review Word Count by LDA Topic',  dpi=300)
    plt.show()

def plot_cloud_by_topic(cloudDf, n_topics, df_type):
    '''Plot word cloud of top N words in each topic'''
    cloud_df = cloudDf.toPandas()
    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    fig, axes = plt.subplots(2, int(n_topics/2), figsize=(n_topics,6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        cloud_df_sub = cloud_df.loc[cloud_df.topic == i]
        topic_words = dict(zip(cloud_df_sub.term, cloud_df_sub.termWeights))
        cloud.generate_from_frequencies(topic_words, max_font_size=500)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER + df_type + 'Word Cloud by LDA Topic', dpi=300, facecolor='white')
    plt.show()


# pyLDAVis
from pyspark.ml.feature import StopWordsRemover,Tokenizer, RegexTokenizer, CountVectorizer, IDF
from pyspark.sql.functions import udf, col, size, explode, regexp_replace, trim, lower, lit
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType, LongType
from pyspark.ml.clustering import LDA


def plot_lda(tokenizedDf, wordVectorizerModel, topicScoresDf, ldaModel):
    '''Plot interactive visualization of pyLDAVis'''
    def format_data_to_pyldavis(df_filtered, count_vectorizer, transformed, lda_model):
        xxx = df_filtered.select((explode(df_filtered.Text_FilteredTokens)).alias("words")).groupby("words").count()
        word_counts = {r['words']:r['count'] for r in xxx.collect()}
        word_counts = [word_counts[w] for w in count_vectorizer.vocabulary]

        data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T, 
                'doc_topic_dists': np.array([x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
                'doc_lengths': [r[0] for r in df_filtered.select(size(df_filtered.Text_FilteredTokens)).collect()],
                'vocab': count_vectorizer.vocabulary,
                'term_frequency': word_counts}
        return data

    def filter_bad_docs(data):
        bad = 0
        doc_topic_dists_filtrado = []
        doc_lengths_filtrado = []

        for x,y in zip(data['doc_topic_dists'], data['doc_lengths']):
            if np.sum(x)==0:
                bad+=1
            elif np.sum(x) != 1:
                bad+=1
            elif np.isnan(x).any():
                bad+=1
            else:
                doc_topic_dists_filtrado.append(x)
                doc_lengths_filtrado.append(y)

        data['doc_topic_dists'] = doc_topic_dists_filtrado
        data['doc_lengths'] = doc_lengths_filtrado

    # format data
    data = format_data_to_pyldavis(tokenizedDf, wordVectorizerModel, topicScoresDf, ldaModel)
    filter_bad_docs(data) # this is, because for some reason some docs apears with 0 value in all the vectors, or the norm is not 1, so I filter those docs.
    return data
