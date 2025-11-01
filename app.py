from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model import EventRecommender
import os

API_KEY = os.getenv("API_KEY", "test123")  # fallback for local testing

app = FastAPI(title="Smart Event Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = EventRecommender()

class UserModel(BaseModel):
    domain: str
    areas_of_interest: List[str]

class EventModel(BaseModel):
    event_name: str
    tags: Optional[List[str]] = []
    skills: Optional[List[str]] = []

class RecommendationRequest(BaseModel):
    user: UserModel
    events: List[EventModel]

@app.post("/recommend")
def recommend_events(request: RecommendationRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    user = request.user.dict()
    events = [event.dict() for event in request.events]
    recommendations = recommender.recommend_events(user, events, top_n=5)
    return {"recommendations": recommendations}

@app.get("/")
def home():
    return {"message": "Smart Event Recommendation API running"}
