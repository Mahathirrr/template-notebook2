# %% [markdown]
# ### Import Libraries

# %%
import googleapiclient.discovery
from google.colab import drive
import numpy as np
import pandas as pd
import re
from IPython.display import Image, display

# !pip install nlp-id
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer

# !pip install nltk
import nltk
# nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
# !pip install wordcloud
from wordcloud import WordCloud
from collections import Counter

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

drive.mount('/content/drive')

# %% [markdown]
# ### List Video

# %% [markdown]
# 1. BELAJAR MACHINE LEARNING MULAI DARIMANA? (https://www.youtube.com/watch?v=aXLVJO7f21c)
# 2. Belajar Machine Learning Dari Awal Buat Yang Ga Jago Matematika (https://www.youtube.com/watch?v=WH1SduDRL_Y)
# 3. 01 - Berkenalan dengan Machine Learning (https://www.youtube.com/watch?v=mEwoAV5_dcA)
# 4. 02 - Klasifikasi Data dengan Decision Tree pada Aplikasi Rapidminer (https://www.youtube.com/watch?v=ovWY9NZXsPA)
# 5. Klasifikasi Data dengan Naive Bayes Classifier pada Aplikasi Rapidminer (https://www.youtube.com/watch?v=7TlVncuh5Lo)
# 6. Klasterisasi Data dengan K-Means pada Aplikasi Rapidminer (https://www.youtube.com/watch?v=ibEC76Emme8)
# 7. SKLearn 01 | Pengenalan Machine Learning | Apa itu Machine Learning | Belajar Machine Learning Dasar (https://www.youtube.com/watch?v=vqgucRK5K1A)
# 8. Kuliah Terbuka #1 - Streaming Magang, Machine Learning & Supervised Learning
#  (https://www.youtube.com/watch?v=4yvBTwbVzsM)
# 9. kaggle 01 | Belajar Exploratory Data Analysis, Visualisasi Data, Klasifikassi | Machine Learning (https://www.youtube.com/watch?v=Op3019SFYzI)
# 10. kaggle 03 | Belajar Machine Learning Dasar dengan Python | Machine Learning Pemula Bahasa Indonesia (https://www.youtube.com/watch?v=DY10uyDy3vQ)

# %% [markdown]
# ### Youtube API Configuration

# %% [markdown]
# Youtube API: https://console.cloud.google.com/apis/api/youtube.googleapis.com/credentials?hl=id&project=turing-clover-440810-e5

# %%
api_key = "AIzaSyAKJlIcRhPF8WwH-gAiUPFjkO8eNshlWFk"  # YouTube API Key
service_name = "youtube"  # API service name
service_version = "v3"  # API version
developer_key = api_key  # Simpen API Key ke dalem variable developer_key

# Bikin YouTube API service object
youtube_client = googleapiclient.discovery.build(
    service_name, service_version, developerKey=developer_key)

# %% [markdown]
# ### Fetch Comments Function

# %%
def fetch_comments(video_id, max_top_comments=50):
    # Request awal untuk ambil komentar dari video based on ID-nya
    initial_request = youtube_client.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100  # Max 100 komentar dalam satu request ke Youtube API
    )

    comment_list = []  # Tampung komen dalam sebuah list

    # Eksekusi request awal
    response = initial_request.execute()

    # Extract komentar dan save ke 'comment_list'
    for item in response['items']:
        comment_data = item['snippet']['topLevelComment']['snippet'] # Top level comment komentar yang langsung ditulis user (bukan hasil reply)
        is_public = item['snippet']['isPublic']
        comment_list.append([
            comment_data['authorDisplayName'], # Nama user yang komen
            comment_data['publishedAt'], # Tanggal dan waktu saat komentar ditulis
            comment_data['likeCount'], # Jumlah likes dari komen tersebut
            comment_data['textOriginal'], # Isi teks asli dari komentar
            comment_data['videoId'], # ID video tempat komentar tersebut ditulis
            is_public # Status apakah komentar bisa dilihat (public) atau tidak
        ])

    """
      Pagination adalah proses membagi data menjadi beberapa halaman. Tujuannya untuk mengambil
      data secara bertahap ketika data terlalu banyak untuk diambil sekaligus dalam satu request

      Misalnya, jika ada 1000 komentar pada video, tetapi API hanya memperbolehkan
      pengambilan 100 komentar per request, maka perlu dilakukan pagination
      untuk mengambil semua komentar tersebut dalam 10 request terpisah

      Pagination dibawah ini menggunakan `nextPageToken` untuk meminta halaman data berikutnya
    """

    # Looping buat mengelola pagination
    while "nextPageToken" in response:
        next_page_token = response['nextPageToken']

        # Request untuk next page
        next_request = youtube_client.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )

        # Execute next request buat dapetin komentar di next page
        response = next_request.execute()

        # Extract komentar dan tambahin ke 'comment_list'
        for item in response['items']:
            comment_data = item['snippet']['topLevelComment']['snippet']
            is_public = item['snippet']['isPublic']
            comment_list.append([
                comment_data['authorDisplayName'],
                comment_data['publishedAt'],
                comment_data['likeCount'],
                comment_data['textOriginal'],
                comment_data['videoId'],
                is_public
            ])

    # Convert 'comment_list' jadi DataFrame
    comments_df = pd.DataFrame(comment_list, columns=['author', 'timestamp', 'likes', 'content', 'video_id', 'public_status'])

    # Sort berdasarkan 'likes' secara descending lalu pilih komentar teratas sebanyak max_top_comments (50)
    top_comments_df = comments_df.sort_values(by='likes', ascending=False).head(max_top_comments)

    return top_comments_df

# %% [markdown]
# Function `fetch_comments` diatas digunakan untuk mengambil semua komentar dari video YouTube tertentu (yang akan di define nanti), mengelola pagination jika jumlah komentarnya banyak, lalu mengurutkannya berdasarkan jumlah likes terbanyak agar lebih relevan untuk analisis nantinya. Hasil akhirnya berupa DataFrame yang berisi daftar komentar yang paling populer.

# %% [markdown]
# ### Execute Fetch Comment Function

# %%
df = pd.DataFrame() # Buat datafram untuk simpen semua komen yang diambil

# List dari semua video IDs
video_ids = [
    'aXLVJO7f21c', 'WH1SduDRL_Y', 'mEwoAV5_dcA', 'ovWY9NZXsPA',
    '7TlVncuh5Lo', 'ibEC76Emme8', 'vqgucRK5K1A', '4yvBTwbVzsM',
    'Op3019SFYzI', 'DY10uyDy3vQ'
]

# Looping untuk tiap video ID lalu ambil komennya
for video_id in video_ids:
    df2 = fetch_comments(video_id)
    df = pd.concat([df, df2], ignore_index=True)

# %%
scrap_path = '/content/drive/MyDrive/UTS TEXT MINING/DataScrapping.csv'
df.to_csv(scrap_path, index=False)

# %%
df.head(10)

# %%
df.tail(10)

# %%
df.sort_values(by='likes', ascending=False)[0:10]

# %%
df.info()

# %%
df['video_id'].value_counts()

# %% [markdown]
# ### Data Cleansing

# %%
def clean_comment(text):
    text = text.lower() # Convert teks ke huruf kecil
    text = text.encode('ascii', 'ignore').decode('ascii') # Remove emoji
    text = re.sub(r'http\S+|www.\S+', '', text) # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text) # Remove tanda baca, angka, dan karakter khusus, jadi cuman nyisain huruf dan spasi
    text = re.sub(r'\b\w\b', '', text) # Remove kata-kata yang terdiri dari 1 karakter aja
    text = re.sub(r'\s+', ' ', text).strip() # Remove spasi extra
    return text

df['cleaned_content'] = df['content'].apply(clean_comment)

# %%
df[['content', 'cleaned_content']].head(10)

# %% [markdown]
# Dari hasil di atas, bisa dilihat kalau proses cleaning udah berhasil. Emoji, tanda baca, URL, dan spasi ekstra udah hilang semua, jadi teksnya lebih rapi dan konsisten dalam huruf kecil. Next step nya, bisa dilanjut dengan **remove stop words** untuk ngurangin kata-kata umum yang nggak terlalu penting biar analisisnya lebih fokus ke kata-kata yang relevan.

# %% [markdown]
# ### Remove Stopwords

# %%
# Buat list stopwords dari library nlp_id
stopword = StopWord()
stopword_list = set(stopword.get_stopword())
print(stopword.get_stopword()[:20])

# %%
def remove_stopwords(text):
    words = text.split()  # Pisahin teks jadi kata-kata
    clean_words = [word for word in words if word not in stopword_list]  # Remove stopwords
    return ' '.join(clean_words)  # Merge lagi kata-kata tsb jadi satu string (teks)

df['stopword_cleaned_content'] = df['cleaned_content'].apply(remove_stopwords)

# %%
df[['cleaned_content', 'stopword_cleaned_content']].head(10)

# %% [markdown]
# Dari hasil di atas, terlihat bahwa proses remove stopwords udah berhasil. Kata-kata umum seperti "saya", "lagi", dan "dan" sudah hilang di kolom `stopword_cleaned_content`, sehingga teksnya lebih fokus pada kata-kata penting. Langkah selanjutnya akan dilakukan **lemmatization** buat ngubah kata-kata menjadi bentuk dasar agar lebih rapi dan konsisten untuk analisis.

# %% [markdown]
# ### Lemmatization

# %% [markdown]
# Lemmatizer adalah proses mengubah kata ke bentuk dasarnya berdasarkan maknanya dalam kalimat. Tujuannya, supaya kata-kata punya bentuk dasar yang konsisten dan lebih mudah dianalisis.

# %%
# Initialize lemmatizer dari library nlp_id
lemmatizer = Lemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

dummy_text = "saya sedang belajar pemrograman dan mendalami machine learning untuk memahami data agar hasilnya lebih baik"
lemmatized_text = lemmatize_text(dummy_text)

print("Teks asli:", dummy_text)
print("Hasil lemmatization:", lemmatized_text)

# %% [markdown]
# Dalam contoh di atas, library `nlp_id` dipilih untuk lemmatization. Sayangnya, library ini masih cukup sederhana, jadi hasilnya kadang mirip dengan **stemming**. Bedanya, stemming cuma memotong akhiran kata tanpa peduli konteks, sehingga kata-kata bisa terpotong terlalu pendek atau jadi nggak natural. Di sini, hasil lemmatization-nya masih agak kasar, misalnya “belajar” jadi “ajar” dan “pemrograman” jadi “program,” padahal konteksnya bisa berubah.

# %%
df['lemmatization_stopword_cleaned_content'] = df['stopword_cleaned_content'].apply(lemmatize_text)

# %%
df[['stopword_cleaned_content', 'lemmatization_stopword_cleaned_content']].tail(10)

# %% [markdown]
# Dari hasil di atas, dapat dilihat kalo proses lemmatization udah berhasil, di mana kata-kata sudah diubah ke bentuk dasarnya di kolom `lemmatization_stopword_cleaned_content`. Next stepnya adalah melakukan **tokenization** agar teks lebih mudah dianalisis saat membuat wordcloud atau analisis frekuensi kata.

# %% [markdown]
# ### Tokenization

# %%
# Gabungin semua data teks jadi satu string
text = ' '.join(df['lemmatization_stopword_cleaned_content'].dropna().values)

# Tokenisasi teks (Pisahkan per kata)
tokens = word_tokenize(text)
tokens[:10]

# %% [markdown]
# ### WordCloud

# %%
# Gabungin lagi tokens menjadi satu string untuk word cloud
tokenized_text = ' '.join(tokens)

wordcloud = WordCloud(width=1300, height=700, background_color='white', collocations=False, random_state=42).generate(tokenized_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# Dari word cloud diatas ini, terlihat jelas kalau beberapa video tentang **machine learning** ini benar-benar menarik perhatian banyak orang.
# - Kata **`bang`** besar banget di sini, yang nunjukin kalau banyak penonton merasa cukup akrab dengan para kreator video, di mana penonton sering memanggil kreator dengan sapaan tersebut.
# - Kemudian, kata **`data`** muncul besar, yang ngga mengherankan karena data adalah fondasi utama di dunia **machine learning**. Banyak banget penonton yang kemungkinan tertarik atau penasaran tentang cara mengolah data, apalagi kalau mereka baru mulai belajar di bidang ini.
# - Selain itu, **`terima kasih`** juga muncul berkali-kali, nunjukin kalo banyak penonton yang merasa terbantu dengan video-video ini. Mereka ngga cuma nonton, tapi benar-benar menghargai usaha kreator dalam menjelaskan konsep yang mungkin awalnya rumit jadi lebih mudah dipahami.
# - Kata **`tutorial`** dan **`video`** mempertegas bahwa konten-konten ini memang bersifat edukatif. Penonton ga cuman dapet penjelasan, tapi juga bimbingan yang praktis. Lalu ada kata-kata seperti **`machine`, `learning`,** dan **`klasifikasi`,** yang menunjukkan topik utama dari video-video ini, yaitu konsep dasar machine learning dan penerapannya, terutama dalam hal klasifikasi data.
# - Kata **`training`** dan **`testing`** juga cukup besar, yang menunjukkan bahwa penonton nggak cuma belajar teori, tapi juga praktek seperti mempersiapkan data untuk **`training`** dan **`testing`** model mereka. Yang menarik lagi, kata **`izin`** juga muncul. Banyak penonton dengan sopan meminta izin untuk menggunakan atau bertanya mengenai materi dari video ini.
# - Kata **`ilmu`** dan **`ajar`** muncul, dimana bisa dibilang kalau video-video ini bukan sekadar tontonan biasa, tapi benar-benar dianggap sebagai sumber ilmu oleh para penontonnya.

# %%
# Hitung frekuensi kemunculan setiap kata dalam tokens
word_counts = Counter(tokens)

# Ambil 20 kata dengan frekuensi tertinggi
top_20_words = word_counts.most_common(20)

print("Top 20 kata dengan frekuensi tertinggi:")
for word, count in top_20_words:
    print(f"{word}: {count}")

# %% [markdown]
# Dari list frekuensi kata di atas, saya bereksperimen membuat **custom stopwords** untuk menghapus kata-kata umum yang kurang relevan dengan topik machine learning. Kata-kata seperti `bang`, `kasih`, `kak`, `bapak`, dsb sering muncul tapi sebenarnya tidak memberikan informasi penting untuk analisis topik ini. Dengan menghapus kata-kata tersebut, word cloud yang dihasilkan bisa lebih fokus pada kata-kata yang benar-benar berkaitan dengan machine learning, sehingga memberikan gambaran yang lebih akurat tentang topik yang dibahas dalam komentar.

# %%
# List kata yang ingin diremove
stopwords_custom = [
    'bang', 'kasih', 'terima', 'terimakasih', 'video', 'izin', 'kak',
    'bapak', 'mohon', 'ga', 'bantu'
]

# Filter tokens untuk delete kata-kata dalam stopwords_custom
filtered_tokens = [word for word in tokens if word not in stopwords_custom]

# Gabungin lagi tokens yang sudah difilter menjadi satu string untuk word cloud
filtered_text = ' '.join(filtered_tokens)

wordcloud_filtered = WordCloud(width=1300, height=700, background_color='white', collocations=False, random_state=42).generate(filtered_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_filtered, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# Dari word cloud diatas setelah dilakukan filter, kelihatan banget kalau kata-kata kayak **`data`**, **`ajar`**, **`machine`**, **`learning`**, dan **`klasifikasi`** jadi topik yang sering dibahas. Ini nunjukin kalau banyak yang fokusnya emang ke pembelajaran soal machine learning, khususnya tentang data dan klasifikasi.
#
# Kata-kata kayak **`jelas`**, **`ilmu`**, **`paham`**, dan **`mudah`** juga cukup besar. Mungkin banyak yang ngarepin penjelasan yang lebih gampang dimengerti dan jelas, atau bisa jadi mereka mengapresiasi konten yang bikin mereka lebih paham. Terus, ada juga kata **`hasil`** dan **`uji`** yang sering muncul, berarti ada ketertarikan buat tau soal pengujian atau hasil model dalam machine learning. Jadi, penonton nggak cuma pengen ngerti teori aja, tapi juga pengen tau prakteknya kayak gimana.
#
# Jadi, intinya word cloud ini nunjukin kalau orang-orang di kolom komentar fokusnya ke belajar machine learning, pengen yang jelas dan mudah, dan ada minat juga ke implementasi praktis kayak pengujian model.

# %%
word_counts = Counter(filtered_tokens)
top_20_words = word_counts.most_common(20)
print("Top 20 kata dengan frekuensi tertinggi:")
for word, count in top_20_words:
    print(f"{word}: {count}")

# %% [markdown]
# ### One-Hot Encoding

# %% [markdown]
# One Hot Encoding adalah metode untuk mengubah kata-kata menjadi representasi angka yang bisa dipahami oleh komputer. Setiap kata unik di teks diwakili oleh vektor yang sebagian besar berisi nol, kecuali satu posisi yang bernilai 1, menunjukkan posisi kata tersebut dalam kosakata. Jadi, setiap kata punya vektor unik yang memisahkannya dari kata lain. Teknik ini membantu dalam memproses teks untuk model machine learning, meskipun tidak menangkap makna atau hubungan antar kata.

# %%
display(Image(filename='/content/drive/MyDrive/UTS TEXT MINING/OneHotEncoder.png'))

# %% [markdown]
# *Source: https://medium.com/@kalyan45/natural-language-processing-one-hot-encoding-5b31f76b09a0*

# %%
# Ubah setiap baris teks di kolom 'lemmatization_stopword_cleaned_content' jadi list kata
tokenized_texts = df['lemmatization_stopword_cleaned_content'].apply(word_tokenize).tolist()
tokenized_texts[:5]

# %%
# Gabung semua kata dari setiap dokumen jadi satu daftar besar (di flatten)
all_words = [word for doc in tokenized_texts for word in doc]
all_words[:10]

# %%
# Reshape Data dengan mengubah 'all_words' jadi array 2D karena OneHotEncoder butuh input 2D
all_words = np.array(all_words).reshape(-1, 1)
all_words

# %%
# Buat encoder dan set sparse_output=False biar hasilnya array penuh (dense), lalu fit ke 'all_words' biar encoder kenal semua kata unik
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(all_words)

# %%
def one_hot_encode_document(doc):
    # Cek jika dokumen kosong
    if not doc:
        # Return vektor nol dengan panjang yang sama seperti kosakata (biar ga error)
        return np.zeros(len(encoder.categories_[0]))  # Disesuaikan berdasarkan ukuran kosakata

    # Reshape setiap kata agar sesuai dengan format input encoder
    encoded_words = encoder.transform(np.array(doc).reshape(-1, 1))
    return encoded_words.mean(axis=0)  # Hitung rata-rata vektor untuk dokumen

# %% [markdown]
# Code diatas melakukan perhitungan rata-rata (mean) vektor one-hot dari semua kata di dokumen dengan tujuan buat nyederhanain representasi tiap dokumen jadi satu vektor yang gampang diolah.
#
# Jadi, kalo nyimpen One-Hot untuk setiap kata, datanya bakal jadi super besar dan penuh dengan angka nol, terutama kalau kosakatanya banyak. Nah, rata-rata  ini bikin tiap dokumen punya vektor dengan panjang tetap, jadi lebih praktis buat input model machine learning yang memang butuh bentuk data seragam. Selain itu, hasil rata-rata ini juga bikin representasi dokumen lebih stabil dan fokus ke makna keseluruhan, bukan cuma ke satu kata aja. Kalau nggak dirata-ratain, data bakal lebih berat dan malah nyusahin saat diproses model.

# %%
# Implemen function ke setiap dokumen
df['one_hot_vector'] = [one_hot_encode_document(doc) for doc in tokenized_texts]

# %%
# Ambil daftar kata unik dari encoder
unique_words = encoder.categories_[0]

one_hot_df = pd.DataFrame(df['one_hot_vector'].to_list(), columns=unique_words)
one_hot_df.head()

# %%
len(one_hot_df.columns)

# %% [markdown]
# Output `one_hot_df` adalah representasi One-Hot Encoding dari tiap komentar dalam bentuk DataFrame. Setiap baris merepresentasikan satu dokumen (komentar), dan setiap kolom mewakili kata unik di seluruh dataset (dengan total 1470 kata unik). Sebagian besar nilai adalah 0 karena proses diatas tadi menghitung rata-rata dari representasi One-Hot setiap kata dalam dokumen, sehingga banyak kata tidak muncul di setiap dokumen dan menghasilkan nilai 0. Angka-angka dalam DataFrame ini menunjukkan frekuensi relatif dari kata-kata dalam dokumen.

# %%
# Menampilkan contoh 3 baris pertama yang memiliki kolom dengan nilai > 0
count = 0
for index, row in one_hot_df.iterrows():
    non_zero_columns = row[row > 0]  # Pilih kolom dengan nilai > 0
    if not non_zero_columns.empty:  # Cuma nampilin kalo ada nilai > 0
        print(f"Baris {index} mengandung kata-kata berikut dengan nilai > 0:")
        print(non_zero_columns)
        print("\n")
        count += 1
    if count == 3:  # Stop looping setelah 3 baris
        break

# %% [markdown]
# Nilai di setiap kolom menunjukkan frekuensi relatif atau kehadiran kata dalam dokumen tersebut. Misalnya, di Baris 0, kata-kata seperti "ai", "ajar", "data", "mantap", dan "science" semuanya punya nilai 0.2. Ini artinya, ketika kita menghitung rata-rata representasi One-Hot di baris tersebut, kata-kata ini muncul dan memberikan kontribusi yang sama.
#
# Semakin tinggi nilainya (mendekati 1), semakin besar kontribusi kata tersebut dalam representasi dokumen.

# %% [markdown]
# Jadi, penggunaan `One-Hot Encoding` di sini tujuannya buat ngubah kata-kata dalam komentar jadi bentuk angka yang lebih gampang dianalisis atau dipakai di model machine learning. Dengan representasi ini, jadinya bisa dilihat pola kemunculan kata di setiap komentar dan bandingin antar-komentar. Jadi, kita bisa dapetin gambaran kata-kata mana yang sering muncul atau penting, yang nantinya bisa bantu buat analisis teks lebih lanjut.

# %% [markdown]
# ### CountVectorizer

# %% [markdown]
# CountVectorizer adalah metode untuk mengubah teks jadi angka dengan menghitung frekuensi kemunculan tiap kata. Setiap kata unik jadi kolom, dan angkanya menunjukkan berapa kali kata itu muncul di tiap dokumen. Teknik ini membantu menganalisis teks secara terstruktur untuk model machine learning.

# %%
display(Image(filename='/content/drive/MyDrive/UTS TEXT MINING/CountVectorization.png'))

# %% [markdown]
# *Source: https://www.researchgate.net/publication/339997426_Opinion-Mining_on_Marglish_and_Devanagari_Comments_of_YouTube_Cookery_Channels_Using_Parametric_and_Non-Parametric_Learning_Models/figures?lo=1&utm_source=google&utm_medium=organic*

# %%
# Panggil CountVectorizer untuk bikin representasi Bag-of-Words (BoW)
vectorizer = CountVectorizer()

bow_matrix = vectorizer.fit_transform(df['lemmatization_stopword_cleaned_content']) # Apply ke text yang udah clean

CountVec = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out()) # Bikin jadi dataframe

# %% [markdown]
# Cara Kerja CountVectorizer:
#
# - CountVectorizer pertama-tama mengidentifikasi semua kata unik di seluruh dokumen dalam kolom `lemmatization_stopword_cleaned_content` dan membangun kosakata dari kata-kata ini. Setiap kata di kosakata ini akan menjadi kolom di matriks Bag-of-Words (BoW).
#
# - Setelah kosakata dibentuk, CountVectorizer menghitung berapa kali setiap kata muncul di masing-masing dokumen. Jika kata tertentu muncul di dokumen, kolom yang sesuai akan berisi angka frekuensi kata tersebut di dokumen itu. Kalau kata tidak muncul, kolomnya akan berisi angka 0.

# %%
CountVec.head()

# %%
len(CountVec.columns)

# %% [markdown]
# Seperti yang terlihat pada gambar diatas, terdapat 1470 kolom yang mewakili kata-kata unik. Setiap baris memiliki nilai 0 atau lebih di setiap kolom tergantung apakah kata tersebut muncul di dokumen atau tidak.

# %%
# Pilih baris pertama
row = CountVec.iloc[50]

# Tampilkan hanya kolom yang memiliki nilai > 0
non_zero_columns = row[row > 0]
print('Data di row Pertama:\n ',df['lemmatization_stopword_cleaned_content'][50])
print("\nNilai TF-IDF nya:")
print(non_zero_columns)

# %% [markdown]
# Jadi, tujuan penggunaan `CountVectorizer` itu buat melihat seberapa sering setiap kata muncul di masing-masing dokumen. Jadi, bisa dapat gambaran tentang kata-kata apa yang paling sering muncul dan pola kata yang ada di kumpulan data. Ini membantu dalam membuat analisis teks lebih lanjut atau untuk model machine learning yang butuh data dalam bentuk angka.

# %% [markdown]
# ### TF-IDF

# %%
display(Image(filename='/content/drive/MyDrive/UTS TEXT MINING/TFIDF2.png'))

# %% [markdown]
# *Source: https://medium.com/@rebirth4vali/tf-idf-from-scratch-with-python-e22033bb99f*

# %% [markdown]
# TF-IDF (Term Frequency-Inverse Document Frequency) adalah cara buat ngukur seberapa pentingnya sebuah kata di dalam dokumen, dibandingkan dengan seluruh dokumen lainnya dalam kumpulan data. Ini berguna biar bisa fokus ke kata-kata yang punya arti lebih spesifik dan ngabaikan kata-kata umum yang sering muncul tapi nggak terlalu informatif.

# %%
display(Image(filename='/content/drive/MyDrive/UTS TEXT MINING/TFIDF.png'))

# %% [markdown]
# *Perhitungan pada Term-Frequencies ada sedikit kesalahan pada gambar, seharusnya 1/3 adalah 0.33. Namun, untuk hasil perkalian TF x IDF sudah benar (bisa dilihat pada table Feature Vectors). Jadi, untuk kasus diatas anggep aja kalo 1/3 adalah 0.66 agar tidak membingungkan.*

# %% [markdown]
# *Source: https://www.researchgate.net/publication/354354484_Automated_Prediction_of_Good_Dictionary_EXamples_GDEX_A_Comprehensive_Experiment_with_Distant_Supervision_Machine_Learning_and_Word_Embedding-Based_Deep_Learning_Techniques/figures?lo=1&utm_source=google&utm_medium=organic*

# %% [markdown]
# Cara kerja TF-IDF dibagi menjadi tiga langkah utama:
#
# 1. **Term Frequency (TF):**
#
#   Mengukur seberapa sering kata muncul dalam setiap dokumen. Misalnya, di pojok kiri atas pada gambar:
#     - Di kalimat pertama ("This is good"), kata "this" muncul 1 kali dari total 3 kata, jadi TF-nya adalah 1/3 = 0.33.
#     - Ini dilakukan untuk setiap kata dalam setiap kalimat.
# 2. **Inverse Document Frequency (IDF):**
#
#   Mengukur seberapa unik atau jarang kata tersebut muncul di seluruh dokumen. Rumusnya pakai logaritma, seperti di bagian bawah gambar.
#   - Di sini, misalnya, kata "good" muncul di 1 dari 3 kalimat, jadi nilai IDF untuk "good" dihitung sebagai log(3/1) ≈ 0.47.
#   - Kata yang sering muncul di banyak dokumen, seperti "this" dan "is", akan memiliki IDF rendah karena kurang informatif. Sedangkan kata yang jarang muncul, seperti "awesome", memiliki IDF tinggi, menunjukkan kata ini lebih spesifik.
# 3. **Mengalikan TF dan IDF untuk Membentuk TF-IDF:**
#
#   Setelah TF dan IDF didapatkan, kalikan keduanya untuk mendapatkan nilai TF-IDF per kata di setiap kalimat.
#
#   - Misalnya, untuk kata "good" di kalimat pertama, TF adalah 0.33 dan IDF adalah 0.47. Jadi, TF-IDF-nya adalah 0.66 * 0.47 ≈ 0.31.
#   - Hasil akhir dari proses ini adalah matriks TF-IDF (feature vectors) yang merepresentasikan seberapa penting setiap kata dalam setiap dokumen.

# %%
# Panggil TfidfVectorizer untuk buat representasi TF-IDF
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatization_stopword_cleaned_content']) # Apply ke text yang udah clean

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()) # Convert ke dataframe

# %%
tfidf_df.head()

# %% [markdown]
# Nilai 0 di banyak kolom pada dataframe diatas itu nunjukin kalo kata-kata tersebut memang ga muncul di dokumen tersebut, jadi nilai TF (Term Frequency)-nya 0. Karena TF-IDF itu didapat dengan mengalikan TF dan IDF, kalau TF-nya 0, hasil akhirnya juga pasti 0, meskipun IDF-nya punya nilai tertentu. Jadi, TF-IDF hanya akan punya nilai lebih dari 0 untuk kata-kata yang memang muncul di dokumen dan dianggap penting.

# %%
# Pilih baris pertama
row = tfidf_df.iloc[0]

# Tampilkan hanya kolom yang memiliki nilai > 0
non_zero_columns = row[row > 0]
print('Data di row Pertama:\n ',df['lemmatization_stopword_cleaned_content'].head(1))
print("\nNilai TF-IDF nya:")
print(non_zero_columns)

# %% [markdown]
# Di atas adalah bukti bahwa di baris pertama hanya kata-kata seperti "ai," "ajar," "data," "mantap," dan "science" yang memiliki nilai TF-IDF lebih dari 0. Ini menunjukkan kalau hanya kata-kata tersebut yang muncul di dokumen pertama, sehingga mereka punya nilai TF yang lebih dari 0. Nilai TF-IDF ini mengindikasikan bahwa kata-kata tersebut dianggap penting dalam konteks dokumen pertama. Kata-kata lain nilainya 0 karena tidak muncul di dokumen ini.

# %% [markdown]
# Jadi, tujuan dari `TF-IDF` ini adalah untuk mengukur seberapa penting suatu kata dalam sebuah dokumen dibandingkan dengan seluruh dokumen lainnya. Dengan TF-IDF, fokus utama bisa lebih ke kata-kata yang benar-benar berarti dan mengabaikan kata-kata umum yang sering muncul tapi kurang informatif. Ini sangat berguna untuk analisis teks atau sebagai input bagi model machine learning.


