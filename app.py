import pickle
import pandas as pd

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from model import get_top_20_recommendations

# -----------------------
# App Initialization
# -----------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -----------------------
# Load Artifacts ONCE
# -----------------------
with open("artifacts/log_reg.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("artifacts/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("artifacts/user_based_pred_df.pkl", "rb") as f:
    user_based_pred_df = pickle.load(f)

with open("artifacts/reviews_df.pkl", "rb") as f:
    reviews_df = pickle.load(f)

# -----------------------
# Routes
# -----------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/recommend", response_class=HTMLResponse)
def recommend(request: Request, username: str = Form(...)):

    # Step 1: Get top 20 recommendations
    top_20 = get_top_20_recommendations(
        username,
        user_item_matrix,
        user_based_pred_df
    )

    if top_20 is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Username not found in dataset"
            }
        )

    # Step 2: Filter reviews for top 20 products
    top_20_products = top_20.index.tolist()
    top_20_reviews = reviews_df[
        reviews_df["name"].isin(top_20_products)
    ].copy()

    # Step 3: Predict sentiment for all reviews
    X_tfidf = tfidf_vectorizer.transform(
        top_20_reviews["cleaned_review"]
    )

    top_20_reviews["predicted_sentiment"] = (
        logistic_model.predict(X_tfidf)
    )

    # Step 4: Compute % positive sentiment per product
    top_5_products = (
        top_20_reviews
        .groupby("name")["predicted_sentiment"]
        .apply(lambda x: (x == "positive").mean() * 100)
        .reset_index(name="positive_review_percentage")
        .sort_values(by="positive_review_percentage", ascending=False)
        .head(5)
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "username": username,
            "top_5": top_5_products.to_dict(orient="records")
        }
    )
