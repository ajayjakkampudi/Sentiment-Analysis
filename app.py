
from distutils.log import debug

from flask import Flask,render_template,request,jsonify,url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
import re

from numpy import vectorize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words=stopwords.words('english')
removing_words=['aren','no','nor','not','ain',"aren't",'don',"don't",'couldn',"couldn't",'didn',"didn't",
                'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',
                "mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",
                'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]

for i in range(0,len(removing_words)):
    if removing_words[i] in stop_words:
        stop_words.remove(removing_words[i])


def text_preproccessing(text):
    text=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
    text=re.sub('@[a-zA-Z]+','',text)
    text=re.sub("[^#a-zA-Z']+",' ',text)
    text=text.lower()
    text=' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split() if word not in stop_words])
    return text


vectorizor=pickle.load(open('vectoriser-Tf-Idf.pkl','rb'))
model=pickle.load(open('sentiment-logistic regression.pkl','rb'))



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text=[]
    text.append(request.form['form'])
    text=vectorizor.transform([text_preproccessing(text[0])])
    predictionValue=model.predict(text)
    result=''
    if predictionValue[0]==1:
        result='Positive'
    elif predictionValue[0]==0:
        result='Negative'
    return render_template('index.html',Prediction='The given statement is {}'.format(result))
    
    



if __name__=="__main__":
    app.run(debug=True)