"""importing libraries"""
from gensim.models import Word2Vec
from openpyxl import *
from nltk.cluster import KMeansClusterer
#import xlsxwriter
import nltk
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn import cluster
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from sklearn import metrics
import nltk

from sklearn.metrics import silhouette_score,pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
nltk.download('stopwords')
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
stop_words = set(stopwords.words('english'))
stop_words2=['canada','toronto','also','ca','ea','ed','ilc']
data=pd.read_excel(r"initialFilteredData.xlsx",index_col=0)
dataDuplicate=data[data.duplicated(subset='description')]

#dataDuplicate.to_excel("duplicates.xlsx")
data=data.drop_duplicates(subset='description',keep='first')
data.to_excel("dataWoDuplicates.xlsx")
num_postings=data.shape[0]
'''Sentence Level Tokenizing'''
finalIndex=[]
finalSentences=[]
start = time.time()
for i in range(num_postings):

    jobDes=(((data.iloc[i,1])))
    sentences=nltk.sent_tokenize(jobDes)
    paragraphs = [p for p in jobDes.split('\n') if p]
    par3=[nltk.sent_tokenize(par) for par in paragraphs]
    newSentences= [item for sublist in par3 for item in sublist]
    index=[data.iloc[i,0] for p in range(len(newSentences))]
    finalSentences.append(newSentences)
    finalIndex.append(index)
    if i==50:
        pass


sent= [item for sublist in finalSentences for item in sublist]
ind= [item for sublist in finalIndex for item in sublist]


data= pd.DataFrame(list(zip(ind, sent,sent)), columns =['id', 'description','sentences'])

print("done sentence level tokenizing")


def preprocess(text):

    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = text.replace("\n", "")
    text=''.join([i for i in text if not i.isdigit()])
    return text
def removeStopWords(text):
    word = [i for i in text if not i in stop_words]
    return word
def removeStopWords2(text):
    word = [i for i in text if not i in stop_words2]
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
data['description']=data['description'].apply(lambda x: preprocess(x.lower()))
tokenizer=RegexpTokenizer(r"\w+")
#tokenizer=word_tokenize()
data['description']=data['description'].apply(lambda x: tokenizer.tokenize(x))
data['description']=data['description'].apply(lambda x: removeStopWords(x))
data['description']=data['description'].apply(lambda x: removeStopWords2(x))
data['description']=data['description'].apply(lambda x: removeSingleLengthWords(x))


data=data[data['description'].map(len)>1]
sentences=data['description']
rawSentences=data['sentences']
rawID=data['id']
print("done preprocessing")
print(len(data['description']))
model = Word2Vec(sentences, min_count=1)


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw


X = []
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))

print("========================")


# note with some version you would need use this (without wv)
#  model[model.vocab]
#print(model[model.wv.vocab])


'''

met=[]
sil=[]
maxC=6
#pD=pairwise_distances(X,metric='euclidean')
for i in range(2,maxC):


    #kmeans=cluster.KMeans(n_clusters=i)
    kmeans = cluster.AgglomerativeClustering(n_clusters=i)
    kmeans.fit(X)

    t1 = time.time()
    sil.append(silhouette_score(X, kmeans.labels_),metric='euclidean')
    t2 = time.time()
    print("for", str(i), ",time required=", str(t2 - t1))
    m=kmeans.inertia_
    met.append(m)
    print("for", str(i), ",time required=", str(t2-t1))
    print("metric",m)



plt.plot(range(2,maxC),met)
plt.plot(range(2,maxC),met,'ro')
plt.title("The Elnow Method")
plt.xlabel("numb of cluster")
plt.ylabel("metric")
plt.show()
plt.close()

plt.plot(range(2,maxC),sil)
plt.title("The silhoutte Method")
plt.xlabel("numb of cluster")
plt.ylabel("metric")
plt.show()
'''

'''
'''

NUM_CLUSTERS = 38
end = time.time()
kclusterer = nltk.cluster.KMeansClusterer(NUM_CLUSTERS,distance=nltk.cluster.util.cosine_distance)

assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
#kmeans=cluster.KMeans(n_clusters=i)
#kmeans.fit(X)
#assigned_clusters=kmeans.labels_
print("done clustering")

end2 = time.time()
print("time",end2-end)

sent=data['description'].apply(lambda x: str(x))
v = TfidfVectorizer(max_features=100)
v.fit(sent)
x_tfidf = v.transform(sent)
features = v.get_feature_names()
X_dense = x_tfidf.todense()

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [[features[i], row[i]] for i in topn_ids]
    feat = [features[i] for i in topn_ids]
    score = [row[i] for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df,feat,score


def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=50):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=50):
    dfs = []
    dfs2 = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df,feat,score = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        c='cluster'+str(label)
        feat=[c,'word']+feat
        score=[c,'score']+score
        dfs.append(feats_df)
        dfs2.append(feat)
        dfs2.append(score)
    return dfs,dfs2
def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), faceacolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=4)
        ax.set_title("cluster = " + str(df.label), fontsize=4)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    plt.savefig("topWords.png")
dfs,dfs2=top_feats_per_cluster(x_tfidf, assigned_clusters, features, 0.1, 50)
#df = pd.DataFrame(dfs[i] for i in range of )
#df=pd.DataFrame(dfs2[0],columns=['Cluster 0','cluster 0'])
dfs2=np.array(dfs2)

dfs2=np.transpose(dfs2)
col=[str(i) for j in range(NUM_CLUSTERS) for i in [j]*2]
df=pd.DataFrame(dfs2,columns=col)
df.to_excel("TopWords"+str(NUM_CLUSTERS)+".xlsx", engine='xlsxwriter')
#plot_tfidf_classfeats_h(top_feats_per_cluster(x_tfidf, assigned_clusters, features, 0.001, 50))


data= pd.DataFrame(list(zip(rawID, rawSentences,assigned_clusters)), columns =['id', 'sentences','Cluster'])

data.to_excel("clusterFinalFixedId_cluster"+str(NUM_CLUSTERS)+".xlsx", engine='xlsxwriter')
#km = cluster.KMeans(n_clusters=NUM_CLUSTERS)

#km.fit(X)
#lab=km.labels_
print("done saving")


#print(assigned_clusters)
"""
end = time.time()
t=end-start
print("TIME REQUIRED",t)
#for index, sentence in enumerate(sentences):
#    print(str(assigned_clusters[index]) + ":" + str(sentence))

kmeans = cluster.MiniBatchKMeans(n_clusters=NUM_CLUSTERS,batch_size=64)
kmeans.fit(X)

labels = kmeans.labels_
print(silhouette_score(X, labels))
#centroids = kmeans.cluster_centers_




#model = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)

print("done tsne")
Y = PCA(n_components=2).fit_transform(X)
print("done tsne2")
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290, alpha=.5)
for j in range(len(sentences)):
    plt.annotate(assigned_clusters[j], xy=(Y[j][0], Y[j][1]), xytext=(0, 0), textcoords='offset points')
plt.show()

N = 1000000
words = list(model.wv.vocab)
fig = go.Figure(data=go.Scattergl(
    x = Y[:, 0],
    y =Y[:, 1],
    mode='markers',
    marker=dict(
        color=np.random.randn(N),
        colorscale='Viridis',
        line_width=1
    ),

))
fig.show()

    #print("%s %s" % (assigned_clusters[j], sentences[j]))


end2 = time.time()
print("time",end2-end)
"""

