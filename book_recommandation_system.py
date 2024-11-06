# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:53:38 2023

@author: Pawshekar
"""

import pandas as pd 
import numpy as np
#load dataset
books=pd.read_csv(r"Books.csv")
ratings=pd.read_csv(r"Ratings.csv")
user=pd.read_csv(r"Users.csv")

#analysis of data
print(books.isnull().sum());
print(user.isnull().sum());
print(ratings.isnull().sum());

print(books.duplicated().sum());
print(user.duplicated().sum());
print(ratings.duplicated().sum());

                                # popularity base recommender system
# consider only book that have min 50 vote and has highest 50 books form with avg rating idea
ratings_with_name=ratings.merge(books,on='ISBN')

#count number of votes on each book
number_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
number_rating_df.rename(columns={'Book-Rating':'num-ratings'},inplace=True)

#count number of rating on each book
avg_rating_df=ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg-ratings'},inplace=True)

# merge above 2 df
popular_df=number_rating_df.merge(avg_rating_df,on='Book-Title')

# keep only those book that has rating >250 and sort in descending
popular_df=popular_df[popular_df['num-ratings']>=250].sort_values('avg-ratings',ascending=False).head(50)

#final popular df which contian avg rating title author num of rating 
popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-ratings','avg-ratings']]


                    #Collaborstive filtering based recommender system

#consider only those user who give minimun 200 votes on books

x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
impt_user=x[x].index

#checking that above important user is in rating_with_name_df and retrive it.
filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(impt_user)]
#books having rating greater then 50
y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index

final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


#displaying pivot table
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)

# find cosine_similarity between for each book with each other books
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores=cosine_similarity(pt) 

def recommend(book_name):
    index=np.where(pt.index==book_name)[0][0]
    #sort the list in decending and it simarity with other books
    data=[]
    similar_item=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    for i in similar_item:
        item=[]
        temp_df=books[books['Book-Title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data
        
        
        
        
        
        
#export and download popular df 
import pickle
pickle.dump(popular_df,open('popular.pkl','wb')) 
pickle.dump(pt,open('pt.pkl','wb')) 
pickle.dump(books,open('books.pkl','wb')) 
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb')) 



        

    






