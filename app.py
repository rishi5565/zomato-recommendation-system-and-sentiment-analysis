# importing all necessary libraries
import streamlit as st
st.set_page_config(page_title="Zomato Recommendation System & Sentiment Analysis", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import gzip, pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading all the saved necessary pickle files
with gzip.open("Pickle Files\\df.pkl", 'rb') as f:
    p = pickle.Unpickler(f)
    df = p.load()
with gzip.open("Pickle Files\\rate_df.pkl", 'rb') as f:
    p = pickle.Unpickler(f)
    rating_df = p.load()
with gzip.open("Pickle Files\\cv.pkl", 'rb') as f:
    p = pickle.Unpickler(f)
    cv = p.load()
with gzip.open("Pickle Files\\clf.pkl", 'rb') as f:
    p = pickle.Unpickler(f)
    classifier = p.load()


# finding the cosine similarity with each other for each point of tfidf matrix (we will be using this score to recommend similar restaurants)
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["sum"])
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(name_unique): # restaurant recommendation function
    idx = df[df["name_unique"] == name_unique].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    sim_idxes = list(score_series[1:6].index)
    simdf = df.iloc[sim_idxes][["name_unique", "rate", "online_order", "book_table"]].reset_index(drop=True)
    return simdf.rename(columns= {"name_unique":"Name", "rate":"Rating", "online_order":"Online Order", "book_table":"Table Booking"})

def get_wordcloud(name_unique): # customer reviews word cloud function
    idx = df[df["name_unique"] == name_unique].index[0]
    name = df.iloc[idx]["name"]
    corpus=rating_df[rating_df['name']==name]['review'].values.tolist()
    corpus=' '.join(x for x in corpus)
    wcld = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
    plt.axis("off")
    plt.title(name)
    plt.imshow(wcld)
    return plt.show()

def get_sentiment(name_unique): # sentiment analysis functions
    ps = PorterStemmer()
    corp = []
    idx = df[df["name_unique"] == name_unique].index[0]
    review = df.iloc[idx]["reviews_list"].replace("Rated","").replace("RATED", "").replace("\\n", "")
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corp.append(review)
    review_transformed = cv.transform(corp).toarray()
    return int(classifier.predict(review_transformed))


title = "Zomato Recommendation System & Sentiment Analysis"
st.title(title)
st.write("First of all, welcome! This is the place where you can search restaurants in Bengaluru and get recommendations of other similar restaurants. You can also see the word cloud and the overall sentiment analysis report of customer reviews.")

name_unique = st.selectbox("Search a Restaurant", df["name_unique"])
idx = df[df["name_unique"] == name_unique].index[0]

details_clicked = st.button("Get Restaurant Details")
recc_clicked = st.button("Get Similar Recommendations")
wcld_clicked = st.button("Get Customer Reviews Word Cloud")
sntmt_clicked = st.button("Get Customer Reviews Sentiment Analysis")

rest_name = str(df.iloc[idx]["name"])
rest_type = str(df.iloc[idx]["rest_type"])
rest_url = str(df.iloc[idx]["url"])
rest_address = str(df.iloc[idx]["address"])
rest_cuisines = str(df.iloc[idx]["cuisines"])
rest_cost = str(df.iloc[idx]["cost"])

if len(df.iloc[idx]["phone"]) == 1:
    rest_phone = str(df.iloc[idx]["phone"][0])
else:
    rest_phone = str(df.iloc[idx]["phone"][0]) + " & " + str(df.iloc[idx]["phone"][1])


if details_clicked == True:
    st.markdown("""
    ## **""" + rest_name + """ [""" + rest_type + """]**

[Restaurant Website](""" + rest_url + """)

**Address:** """ + rest_address + """

**Phone:** """ + rest_phone + """

**Cuisines:** """ + rest_cuisines + """

**Approximate Cost for two:** Rs. """ + rest_cost + """
    """)

if recc_clicked == True:
    simdf = recommend(name_unique)
    st.success("Recommendations Fetched Successfully!")
    st.table(simdf)

if wcld_clicked == True:
    with st.spinner("Fetching..."):
        try:
            wcld = get_wordcloud(name_unique)
            st.success("Word Cloud Fetched Successfully!")
            st.pyplot(wcld)
        except:
            st.error("Oops! Insufficient number of customer reviews. Please try another restaurant.")

if sntmt_clicked == True:
    sentiment = get_sentiment(name_unique)

    if sentiment == 0:
        with st.spinner("Analysing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i+1)
            st.success("Customer Reviews Sentiment Analysed Successfully!")    
            st.subheader("Sentiment Analysis Report: Negative")


    if sentiment == 1:
        with st.spinner("Analysing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i+1)
            st.success("Customer Reviews Sentiment Analysed Successfully!")    
            st.subheader("Sentiment Analysis Report: Neutral")


    if sentiment == 2:
        with st.spinner("Analysing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i+1)
            st.success("Customer Reviews Sentiment Analysed Successfully!")    
            st.subheader("Sentiment Analysis Report: Positive")