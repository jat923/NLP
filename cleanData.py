import xlrd
from openpyxl import *
import re
import string
import numpy
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
stop_words = set(stopwords.words('english'))
data=pd.read_excel(r"initialFilteredData.xlsx",index_col=0)
data=data.drop_duplicates(subset='description',keep='first')
print(data.count())
def preprocess(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = text.replace("\n", "")
    return text
def removeStopWords(text):
    word = [i for i in text if not i in stop_words]
    return word
def removeSingleLengthWords(text):
    text = [i for i in text if not len(i) <= 1]
    return text
def partsOfSpeech(text):
    tagged=nltk.pos_tag(text)
    taggedFiltered = [word for word, tag in tagged
                    if tag in ('RB','RBR','RBS','JJR','JJS','JJ')]
    return taggedFiltered
lemmatizer = WordNetLemmatizer()
data['description']=data['description'].apply(lambda x: lemmatizer.lemmatize(x.lower()))

tokenizer=RegexpTokenizer(r"\w+")
#tokenizer=word_tokenize()
data['description']=data['description'].apply(lambda x: tokenizer.tokenize(x))
data['description']=data['description'].apply(lambda x: removeStopWords(x))


data['description']=data['description'].apply(lambda x: removeSingleLengthWords(x))
personalityTraitsKeyword1=['self disciplined', 'self controlled', 'self mastery', 'problem solving skills', 'willing', 'ability', 'determination', 'persistence', 'flexibility', 'work ethic', 'technical competency', 'honesty', 'communication skills']
keys = pd.read_excel(r"RawData/rawKeyword.xls", sheet_name="Sheet1")
personalityTraitsKeyword2 = list(keys.iloc[:, 0])
personalityTraitsKeyword2=personalityTraitsKeyword2+['active', 'adaptive', 'affability', 'affectionate', 'alert', 'ambitious', 'attentive', 'balanced', 'benevolent', 'careful', 'characterful', 'charitable', 'creative', 'compassionate', 'confident', 'considerate', 'cooperative', 'courageous', 'curious', 'dependable', 'determined', 'diligent', 'disciplined', 'dispassionate', 'dutiful', 'encouraging', 'energetic', 'enthusiastic', 'excellent', 'faithful', 'flexible', 'forgiving', 'friendly', 'frugal', 'generous', 'gritty', 'hard-working', 'harmonious', 'honest', 'honourable', 'hopeful', 'humble', 'independent', 'industrious', 'integrous', 'initiative', 'just', 'kind', 'liberal', 'listening', 'lively', 'logical', 'loving', 'loyal', 'merciful', 'methodical', 'mindful', 'moderate', 'modest', 'nea', 'open-minded', 'orderly', 'organised', 'passionate', 'patient', 'persistent', 'polite', 'pragmatic', 'prudent', 'punctual', 'purposeful', 'quality','rational', 'reasonable', 'reliable', 'resolute', 'respectful', 'righteous', 'silent', 'sincere', 'simple', 'stable', 'steadfast', 'strong', 'supportive', 'temperate', 'thrifty', 'tidy', 'truthful', 'trustworthy', 'unselfish', 'valiant', 'vital', 'warm', 'wise']

lemmatizer = WordNetLemmatizer()

ptkeys2Dict={lemmatizer.lemmatize(i.lower()):i for i in personalityTraitsKeyword2}
ptkeys2Set= set([lemmatizer.lemmatize(i.lower()) for i in personalityTraitsKeyword2])

ptkeys1Dict={lemmatizer.lemmatize(i):i for i in personalityTraitsKeyword1}
ptkeys1Set= set([lemmatizer.lemmatize(i) for i in personalityTraitsKeyword1])
ptList=[]
data.to_excel("cleanData2.xlsx")
num_postings=data.shape[0]
for i in range(num_postings):
    jobDes=set(((data.iloc[i,1])))

    ptWords=ptkeys2Set.intersection(jobDes)

    j=0
    for i in personalityTraitsKeyword1:
        if i in jobDes:
            ptList.append(i)
            j += 1

    if len(ptWords) == 0 and j == 0:
        ptList.append('no personal traits required')
    ptList+=list(ptWords)
    #print(ptList)
#data['filtered']=data['description'].apply(lambda x: partsOfSpeech(x))

#print(data.head())
#des=data['description'].str.split("\n",n=2,expand=True)
#print(des.head())



data.to_excel("cleanData2.xlsx")
dfPersonalTraits = pd.DataFrame(data={'cnt': ptList})
dfPersonalTraitsTop50 = dfPersonalTraits['cnt'].value_counts().reset_index().rename(columns={'index': 'traits'}).iloc[:100]

#dfPersonalTraits = dfPersonalTraits.replace(tool_keywords1_dict)
layout = dict(
    title='Personality Traits',
    yaxis=dict(
        title='% of job postings',
        tickformat=',.0%',
    )
)

fig = go.Figure(layout=layout)
fig.add_trace(go.Bar(
    x=dfPersonalTraitsTop50['traits'],
    y=dfPersonalTraitsTop50['cnt']/num_postings
))

fig.show()
