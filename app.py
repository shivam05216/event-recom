from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="Event Recommendation API")

# Load dataset
events_df = pd.read_csv("model/events.csv")

# Combine text features
events_df["combined"] = (
    events_df["Event_Name"].fillna("") + " " +
    events_df["Tags"].fillna("") + " " +
    events_df["Domain"].fillna("")
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(events_df["combined"])

class UserData(BaseModel):
    interests: List[str]
    domain: List[str]

@app.post("/recommend")
async def recommend_events(data: UserData):
    try:
        user_text = " ".join(data.interests + data.domain)
        user_vec = vectorizer.transform([user_text])
        similarity = cosine_similarity(user_vec, tfidf_matrix)
        top_indices = np.argsort(similarity[0])[::-1][:5]

        recommended = events_df.iloc[top_indices][["Event_Name", "Tags", "Domain"]].to_dict(orient="records")
        return {"recommended_events": recommended}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
