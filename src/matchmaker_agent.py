from src.utils import client, vision_model
import json

def matchmaker_agent(query_metadata, retrieved_items_metadata):
    prompt = f"""
You are a clothing stylist assistant. Suggest best matching clothes.
User query: {query_metadata}

Candidate items: {retrieved_items_metadata}

Recommend top 2-3 matches and explain why.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content


