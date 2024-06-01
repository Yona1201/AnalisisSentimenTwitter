import streamlit as st
import pandas as pd
import joblib
from preprocessing import clean_text, case_folding, tokenize, remove_stopwords, stem_text, normalisasi


model = joblib.load('svm.pkl')
vectorizer = joblib.load('tfidf.pkl')

st.title('Sentiment Analysis Gojek with SVM + TFIDF')

# Pilihan input: Teks atau File
option = st.sidebar.selectbox('Choose Input Option:', ['Text', 'File'])

if option == 'Text':
    # Input teks
    user_input = st.text_area('Enter Text:', '')
    if st.button('Analyze'):
        # Preprocessing teks
        cleaned_text = clean_text(user_input)
        normalised_text = normalisasi(cleaned_text)
        folded_text = case_folding(normalised_text)
        tokenized_text = tokenize(folded_text)
        wstopword_text = remove_stopwords(tokenized_text)
        stemmed_text = ' '.join(stem_text(wstopword_text))


        # Tampilkan tahapan preprocessing
        st.subheader('Preprocessing Steps:')
        st.write(pd.DataFrame({'Step': ['Cleaning','Normalisasi' ,'Case Folding', 'Tokenization', 'Stopword Removal', 'Stemming'],
                               'Result': [cleaned_text, normalised_text, folded_text, tokenized_text,  wstopword_text,
                                          stemmed_text]}))

        # Proses teks dan lakukan prediksi
        new_data_tfidf = vectorizer.transform([stemmed_text])
        prediction = model.predict(new_data_tfidf)
        st.write('Sentiment:', prediction)

elif option == 'File':
    # Input file Excel/CSV
    uploaded_file = st.file_uploader('Upload Excel/CSV File:', type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Baca file dan lakukan prediksi
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)

        # Preprocessing teks dalam kolom 'Text'
        df['text'] = df['text'].astype(str)
        df['Cleaned'] = df['text'].apply(clean_text)
        df['Normalized'] = df['Cleaned'].apply(normalisasi)
        df['Case Folded'] = df['Cleaned'].apply(case_folding)
        df['Tokenized'] = df['Case Folded'].apply(tokenize)
        df['Stopword Removal'] = df['Tokenized'].apply(remove_stopwords)
        df['Stemmed'] = df['Stopword Removal'].apply(stem_text)
        df['Stemmed'] = df['Stemmed'].apply(lambda x: ' '.join(x))

        # Proses teks dan lakukan prediksi
        # Jumlah fitur yang diinginkan sesuai dengan model SVM
        new_data_tfidf = vectorizer.transform(df['Stemmed'])

        predictions = model.predict(new_data_tfidf)
        df['Sentiment'] = predictions

        # Tampilkan jumlah label
        st.subheader('Sentiment Distribution:')
        st.write(df['Sentiment'].value_counts())

        # Tampilkan barchart
        st.bar_chart(df['Sentiment'].value_counts())

        # Tampilkan tahapan preprocessing
        st.subheader('Preprocessing Steps:')
        st.write(df)
