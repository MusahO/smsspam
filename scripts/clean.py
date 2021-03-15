# remove stopwords..punct etc
import re
import string
from nltk.corpus import stopwords

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