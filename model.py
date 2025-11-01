# model.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EventRecommender:
    def __init__(self):
        # Load a semantic embedding model (lightweight but powerful)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def recommend_events(self, user, events, top_n=5):
        """
        Recommend events based on semantic similarity between
        user interests and event tags.
        """
        user_profile = " ".join([user.get("domain", "")] + user.get("areas_of_interest", []))
        user_vector = self.model.encode([user_profile])

        recommendations = []

        for event in events:
            event_text = " ".join(
                [event.get("event_name", "")] +
                (event.get("tags") or []) +
                (event.get("skills") or [])
            )
            event_vector = self.model.encode([event_text])
            similarity = cosine_similarity(user_vector, event_vector)[0][0]

            recommendations.append({
                "event_name": event.get("event_name"),
                "tags": event.get("tags"),
                "skills": event.get("skills"),
                "similarity_score": float(similarity)
            })

        # Sort by similarity (highest first)
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top N
        return recommendations[:top_n]
