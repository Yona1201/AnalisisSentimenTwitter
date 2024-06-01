import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

#Cleaning
def clean_text(tweet):
    # Menghapus URL
    url = re.compile(r'https?://\s+|www\.\s+')
    tweet = url.sub('', tweet)

    # Menghapus HTML
    html = re.compile(r'<.*?>')
    tweet = html.sub('', tweet)

    # Menghapus emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticon
                               u"\U0001F300-\U0001F5FF"  # Simbol & Piktogram
                               u"\U0001F680-\U0001F6FF"  # Transportasi & Simbol
                               u"\U0001F1E0-\U0001F1FF"  # Bendera (kewarganegaraan)
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub('', tweet)

    # Menghapus angka
    tweet = re.sub(r'\d+', '', tweet)

    # Menghapus simbol
    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)  # Menghapus semua simbol kecuali huruf, angka, dan spasi

    return tweet

#Normalisasi
norm = {
        'gmn': 'bagaimana',
        'km': 'kamu',
        'skrg': 'sekarang',
        'sblmnya': 'sebelumnya',
        'bls': 'membalas',
        'trs': 'terus',
        'tp': 'tapi',
        'krn': 'karena',
        'gak': 'tidak',
        'yg': 'yang',
        'sm': 'sama',
        'utk': 'untuk',
        'aja': 'saja',
        'nggak': 'tidak',
        'dmn': 'di mana',
        'dr': 'dari',
        'mantaap':'mantap',
        'mantaaap':'mantap',
        'mantappp':'mantap',
        'bgs ': 'bagus',
        'bgus': 'bagus',
        'jd': 'jadi',
        'yg': 'yang',
        'jelekkk':'jelek',
        'jelekk':'jelek',
        'bgt':'banget',
        'bangettt':'banget',
        'bangett':'banget',
        'skrg' :'sekarang',
        }

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

#Case Folding
def case_folding(text):
    return text.lower()

#Tokenization
def tokenize(text):
    tokens = text.split()
    return tokens

#Stopword Removal
#Memakai Library NLTK
stop_words = stopwords.words('indonesian')
def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

#Stemming
#Memakai library Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_text(text):
    return [stemmer.stem(word) for word in text]
