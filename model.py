def get_top_20_recommendations(username, user_item_matrix, user_based_pred_df):
    """
    Returns top 20 product recommendations for a given user
    using precomputed user-based collaborative filtering results.
    """

    if username not in user_item_matrix.index:
        return None

    # Predicted ratings for the user
    user_preds = user_based_pred_df.loc[username]

    # Remove products already rated by the user
    rated_products = user_item_matrix.loc[username].dropna().index
    user_preds = user_preds.drop(rated_products)

    # Top 20 recommendations
    top_20 = user_preds.sort_values(ascending=False).head(20)

    return top_20
