## **Project Introduction**

In this project I'll be exploring the Zomato Bengaluru dataset to extract insights and then will be building a restaurant recommendation system with that information. I'll also be building a sentiment analysis model based on the customer reviews that will give an overall sentiment of the customers to the end user. Finally, I'll be packing everything into a web-app using Streamlit and host everything to cloud for anyone to use.

**WEB-APP LINK:** [Zomato Recommendation System & Sentiment Analysis App](https://share.streamlit.io/rishi5565/zomato-recommendation-system-and-sentiment-analysis/main/app.py)


## **Project Overview**

* Built Zomato restaurant recommendation system & customer review sentiment analysis for Bengaluru city from scratch and packed into web-app using Streamlit and deployed to cloud for anyone to use.
* Cleaned, explored and manipulated the entire data extensively on Python to make it usable for our use case.
* Engineered new features as and when necessary in accordance with our project requirements.
* Explored and extracted meaningful insights from the data using various methods ( Bar plots, scatter and KDE plots, word clouds, folium maps, etc. )
* Used Geospatial Mapping to visualize Restaurant Density around Bengaluru.
* Used Term frequency–Inverse Document Frequency (TF-IDF) to create matrix of relevant words and computed cosine similarities between the them to use in our recommendation system.
* Performed under-sampling to balance classes before making sentiment analysis model.
* Performed multi-class classification with an accuracy of 82% using Multinomial Naive Bayes Classifier to build the customer review sentiment analysis model.
* Hyper-parameter tuned our sentiment analysis model to get the best performance.
* Used Streamlit to build front-end interface of the recommendation system and sentiment analysis web-app.

## **Potential Uses**

* Users who want to find similar restaurants to the ones they already go to.
* To compare prices, cuisines, etc. of similar restaurants to get a better idea.
* Restaurant owners to better adjust their prices and quality according to similar restaurants.
* Zomato to improvise their service and optimize deliveries.
* Users to decide a restaurant based on an auto generated sentiment analysis report.
* Gain competitive edge over rival restaurants.


## **Exploratory Data Analysis (EDA)**
* Top 20 Restaurant chains in Bengaluru:![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/1.png)

* Availability of Online Order:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/2.png)

* Availability of Table Booking:![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/3.png)

* Restaurant ratings distribution:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/4.png)

* Rating and Approx Cost Relationship scatter plot:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/5.png)
We see that it is not easy to visually interpret the data as a lot of points are overlapping on top of each other. Hence, we decide to use a kernel density estimate (KDE) plot to better visually interpret the information.
* KDE Plot:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/6.png)
Here, we can see that majority of the ratings lie between 3.5 to 4.0 and in the cost range of 0 to 1000.
* Approx Cost Distribution:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/7.png)
We see that the distribution in right skewed and leptokurtic. This is expected as our cost starts from after the point 0 and there tends to be less and less restaurants with very high costs.

* Top 20 Restaurant types in Bengaluru:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/8.png)
We see that Quick Bites is the most popular restaurant type. This makes sense as Bengaluru is a very busy city of working professionals who don't have much time so they prefer to make use of Quick Bites.
* Top 20 Most Popular Cuisines:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/9.png)
We see that North Indian is the most popular cuisine. This makes some sense as majority of working professionals in Bengaluru hail from North India.

* Top 20 Locations with most Restaurants in Bengaluru:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/10.png)

* Geospatial Mapping of Restaurant Density around Bengaluru:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/11.png)
We see that South of Bengaluru has a high density of Restaurants. A higher density can also be seen at the Central area of the city.

* Word Cloud of dishes liked by customers for top 9 restaurant types:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/12.png)

* Word Cloud of customer reviews for top 9 restaurant chains:
![enter image description here](https://github.com/rishi5565/zomato-recommendation-system-and-sentiment-analysis/raw/main/EDA%20Images/13.png)


## **Data Cleaning & Pre-Processing:**
* Removed insignificant features after analyzing descriptive stats.
* Engineered new features from existing features in accordance with our project objective.
* Simplified complex features to better fit our model.
* Pre-processed each and every feature in accordance of finding the best cosine similarities for our recommendation system.
* Used Term frequency–Inverse Document Frequency (TF-IDF) to create matrix of relevant words and computed cosine similarities between the them to use in our recommendation system.

## **Sentiment Analysis Model Building:**
* After classifying all the user ratings with a custom function, we realized that our classes are imbalanced in an approximate ratio of 4:1:1. We decided to use under-sampling to balance our classes because the total amount of user reviews data was huge and we needed only a fraction of it to train the classifier model effectively.
* Performed stemming of all review sentences from the under-sampled data and removal of stopwords to build our sentiment analysis model.
* Used Count Vectorizer to fit transform the corpus and split the data into training and testing sets.
* *We decide to use **Multinomial Naive Bayes classifier** building our Sentiment Analysis model as it is very suitable for classification with discrete features (e.g., word counts for text classification). Also it is easy to implement and understand.*
* We performed hyper-parameter tuning of alpha on the model to improve the performance and were able to achieve an accuracy score of 82%.

**We compressed and saved all the required objects as pickle files to be used further in our Streamlit web-app.**

## **Conclusion**
We were able to build the front end of the app successfully using Streamlit. We defined all the functions with all the necessary conditions and then proceeded to deploy the app on cloud for anyone to use.
**WEB-APP LINK:** [Zomato Recommendation System & Sentiment Analysis App](https://share.streamlit.io/rishi5565/zomato-recommendation-system-and-sentiment-analysis/main/app.py)

Data Source: [Link](https://www.kaggle.com/datasets/absin7/zomato-bangalore-dataset)

### [Note]:  _**Please refer to the Project Notebook file for all the detailed in-depth information regarding this project. The above information is just a brief summary of the project.**_

Thank You,

Rishiraj Chowdhury ([rishiraj5565@gmail.com](mailto:rishiraj5565@gmail.com))


