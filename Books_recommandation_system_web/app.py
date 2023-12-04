from flask import Flask,render_template,request
import pandas as pd
import numpy as np

popular_df=pd.read_pickle(open('popular.pkl','rb'))
pt=pd.read_pickle(open('pt.pkl','rb'))
books=pd.read_pickle(open('books.pkl','rb'))
similarity_scores=pd.read_pickle(open('similarity_scores.pkl','rb'))

app=Flask(__name__,template_folder='Template')

@app.route('/')
def index():
          return render_template('index.html',
                                 book_name=list(popular_df['Book-Title'].values),
                                 author=list(popular_df['Book-Author'].values),
                                 image=list(popular_df['Image-URL-M'].values),
                                 votes=list(popular_df['num-ratings'].values),
                                 rating=list(popular_df['avg-ratings'].values)
                                 )
          

@app.route('/recommend')
def recommend_ui():
          return render_template('Recommand.html')
@app.route('/recommend_books',methods=['post'])
def recommend():
          user_input=request.form.get('search')
          index=np.where(pt.index==user_input)[0][0]
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
          return render_template('Recommand.html',data=data)

if __name__=='__main__':
          app.run(debug=True)
