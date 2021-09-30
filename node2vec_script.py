from node2vec import Node2Vec
import networkx as nx
import pandas as pd

df = pd.read_csv('C:/Users/dorgo/Downloads/browsi_data.csv')

graph = nx.from_pandas_edgelist(df, 'site_id', 'user_id', ['site_category'])

node2vec = Node2Vec(graph, dimensions=10, walk_length=16, num_walks=20)

model = node2vec.fit(window=10, min_count=1)

sites = pd.Series(list(set(df.site_id)))

lis=[]
for i in set(df.site_id):
    a = model.wv.get_vector(i)
    lis.append(a)

site_emb = pd.DataFrame(lis)
site_emb = pd.concat([site_emb, sites],axis=1)
site_emb = site_emb.rename(columns=lambda s: 'emb' + str(s))

cols=pd.Series(site_emb.columns)
for dup in cols[cols.duplicated()].unique():
    cols[cols[cols == dup].index.values.tolist()] = \
        [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]

site_emb.columns = cols
categories = df[['site_id', 'site_category']].drop_duplicates()
site_embbeding = pd.merge(site_emb, categories, left_on='emb0.1', right_on='site_id')


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
site_embbeding['target'] = le.fit_transform(site_embbeding.site_category)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = site_emb.filter(regex="emb\d$")
y = site_emb.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test,y_pred)


