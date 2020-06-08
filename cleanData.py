import xlrd
from openpyxl import *
import re
import string
import numpy
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
data=pd.read_excel(r"initialFilteredData.xlsx",index_col=0)

print(data.count())
#print(data[data.duplicated(['description'], keep=False)].count())
data=data.drop_duplicates(subset='description',keep='first')
print(data.count())
num=0
for i in range(data.shape[0]):
    num=num+1
    text=(data.iloc[i,1]).lower()
    a=text
    print(text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = text.replace("\n", "")
    lemmatizer = WordNetLemmatizer()
    #stemmer=PorterStemmer()
    text = word_tokenize(text)
    text1=[lemmatizer.lemmatize(word) for word in text]
    #text1 = [stemmer.stem(word) for word in text]

    text2 = [i for i in text1 if not i in stop_words]
    text3=[i for i in text2 if not len(i)<=1]
    data.iloc[i,1]=str(text3)


data.to_excel("cleanData.xlsx")


