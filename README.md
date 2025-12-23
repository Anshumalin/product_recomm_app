# product_recomm_app
This project presents an end-to-end sentiment-aware recommendation system designed for an e-commerce platform.The goal of the system is to recommend products to users not only based on their historical preferences but also by incorporating user sentiment expressed in product reviews, thereby improving the quality and reliability of recommendations.

The project is divided into two major components: sentiment analysis and product recommendation, which are integrated into a single deployable application.

🔹 Sentiment Analysis

User reviews are preprocessed using standard NLP techniques such as lowercasing, stopword removal, lemmatization, and noise removal. Features are extracted using TF-IDF vectorization, capturing both unigrams and bigrams. Multiple machine learning models were evaluated, including Logistic Regression, Naive Bayes, and Random Forest.
Based on performance—especially recall and F1-score for the minority (negative) class—Logistic Regression was selected as the final sentiment classifier. This ensures that products with poor reviews are accurately identified and filtered out.

🔹 Recommendation System

A user-based collaborative filtering approach is used to generate personalized product recommendations. A user–item rating matrix is constructed from historical ratings, and cosine similarity is applied to identify similar users. Predicted ratings are computed using a weighted average of ratings from similar users.
For a given user, the system recommends the top 20 products the user is most likely to purchase.

🔹 Sentiment-Based Fine-Tuning

To enhance recommendation quality, sentiment analysis is applied to all reviews of the top 20 recommended products. For each product, the percentage of positive reviews is calculated, and the top 5 products with the highest positive sentiment are finally recommended to the user.

🔹 Deployment

The entire system is deployed as a web application using FastAPI. All trained models and precomputed artifacts are serialized using pickle and loaded at runtime, ensuring no retraining is required on the server. The application allows users to enter a username and receive real-time, sentiment-aware product recommendations through a simple and interactive UI.

🚀 Key Technologies

Python, Pandas, NumPy

Scikit-learn (TF-IDF, Logistic Regression)

Collaborative Filtering (Cosine Similarity)

FastAPI, HTML/CSS

Cloud Deployment (Render)

This project demonstrates the practical application of machine learning, NLP, and software deployment to build a scalable and user-focused recommendation system.
