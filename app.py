from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Model ve TF-IDF vektörleştiriciyi yükleme
with open('model.pk1', 'rb') as model_file:
    pac = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

def predict_news(news_text):
    news_tfidf = tfidf_vectorizer.transform([news_text])
    prediction = pac.predict(news_tfidf)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    prediction = predict_news(news_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
