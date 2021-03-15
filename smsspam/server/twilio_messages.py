import os
import json
from flask import Flask, jsonify, make_response
from flask_cors import CORS
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()  # load environment variables

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
twilio_api = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)



# https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    
    STOPWORDS = stopwords.words('english') + ['é', 'u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return " ".join([word for word in text.split() if word.lower() not in STOPWORDS])

def get_messages():
    sms = list(twilio_api.messages.stream())

    messages = []
    for txt in sms:
        txt = clean_text(txt.body)
        messages.append(txt)
    return messages

app = Flask(__name__)
CORS(app)

@app.route('/messages')
def messages():
    return make_response(jsonify({'messages': get_messages()}), 200)
