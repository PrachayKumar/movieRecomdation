import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Wiki_Telugu_Movies_1930_1999.csv")
df

df1 = df.fillna(" ")
df1.info()

df1["update"] = df1["Director"]+" "+df1["Cast"]+" "+df1["Genre"]+" "+df1["Music Composer"]

df1["update"]

cv = CountVectorizer()

vec = cv.fit_transform(df1["update"] )

sim = cosine_similarity(vec)

usermovie = input("Enter movie :")

ind=df1[df1.Title==usermovie].index[0]
ns = list(enumerate(sim[ind]))
s = sorted(ns,key=lambda x:x[1],reverse=True)

for i in range(5):
  print(df1["Title"].loc[s[i][0]])
