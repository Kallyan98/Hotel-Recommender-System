import re
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from huggingface_hub import InferenceClient

# -------------------
# Auto-download NLTK resources if missing
# -------------------
nltk_resources = {
    "punkt": "tokenizers",
    "punkt_tab": "tokenizers",
    "averaged_perceptron_tagger_eng": "taggers"
}

for resource, folder in nltk_resources.items():
    try:
        nltk.data.find(f"{folder}/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# -------------------
# Hugging Face Inference Client
# -------------------
HF_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # replace with your token
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -------------------
# Dummy hotel database (prices in INR)
# -------------------
HOTELS = [
    {"name": "Sea Breeze Resort", "price_adult": 4800, "price_child": 650, "currency": "INR", "rating": 3.7, 
     "location": "Goa", "amenities": ["sea front"], "sightseeing": ["Baga Beach", "Fort Aguada"]},

    {"name": "Lakeview Serenity Hotel", "price_adult": 4200, "price_child": 600, "currency": "INR", "rating": 4.2, 
     "location": "Udaipur", "amenities": ["lake view"], "sightseeing": ["City Palace", "Lake Pichola"]},

    {"name": "Budget Stay Central", "price_adult": 2500, "price_child": 500, "currency": "INR", "rating": 3.0, 
     "location": "New Delhi", "amenities": ["city center"], "sightseeing": ["India Gate", "Connaught Place"]},

    {"name": "Mountain Escape Lodge", "price_adult": 3500, "price_child": 550, "currency": "INR", "rating": 4.1, 
     "location": "Manali", "amenities": ["mountain view"], "sightseeing": ["Solang Valley", "Hidimba Temple"]},

    {"name": "Royal Heritage Palace", "price_adult": 5000, "price_child": 700, "currency": "INR", "rating": 4.5, 
     "location": "Jaipur", "amenities": ["luxury", "heritage"], "sightseeing": ["Amber Fort", "Hawa Mahal"]},
]

# -------------------
# Extract user preferences
# -------------------
def extract_preferences(prompt: str):
    tokens = word_tokenize(prompt)
    pos_tag(tokens)  # POS tagging (English)

    adults, children, budget, check_in, check_out = 1, 0, None, None, None

    m = re.search(r"(\d+)\s*adults?", prompt, re.IGNORECASE)
    if m: adults = int(m.group(1))

    m = re.search(r"(\d+)\s*children?", prompt, re.IGNORECASE)
    if m: children = int(m.group(1))

    m = re.search(r"budget\s*(?:under|below)?\s*\$?(\d+)", prompt, re.IGNORECASE)
    if m: budget = int(m.group(1))

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", prompt)
    if len(dates) >= 2:
        check_in, check_out = dates[0], dates[1]

    return {"adults": adults, "children": children, "budget": budget,
            "check_in": check_in, "check_out": check_out}

# -------------------
# Day-wise breakdown
# -------------------
def daywise_breakdown(hotel, check_in, check_out, adults, children):
    start = datetime.strptime(check_in, "%Y-%m-%d")
    end = datetime.strptime(check_out, "%Y-%m-%d")
    nights = (end - start).days

    # ✅ Ensure at least 1 night
    if nights <= 0:
        nights = 1  

    breakdown = []
    total_fare = 0

    for i in range(nights):
        day = start + timedelta(days=i)
        # ✅ Use correct fixed adult/child prices
        price_adult = hotel["price_adult"]
        price_child = hotel["price_child"]

        # simulate weekend price increase
        if day.weekday() >= 5:  # Saturday (5) or Sunday (6)
            price_adult = int(price_adult * 1.2)
            price_child = int(price_child * 1.2)

        nightly_total = (adults * price_adult) + (children * price_child)
        total_fare += nightly_total

        breakdown.append({
            "Date": day.strftime("%Y-%m-%d"),
            "Price per Adult": price_adult,
            "Price per Child": price_child,
            "Total for Night": nightly_total,
            "Currency": hotel["currency"]
        })

    return pd.DataFrame(breakdown), total_fare
# -------------------
# Llama recommendation
# -------------------
def llama_recommendation(user_query, hotels, fares, prefs):
    hotel_text = "\n".join(
    [f"{h['name']} - Adult Price: {h['price_adult']} {h['currency']}, "
     f"Child Price: {h['price_child']} {h['currency']}, rating {h['rating']}, "
     f"amenities: {h['amenities']}, sightseeing: {h['sightseeing']}, "
     f"Total Fare: {fares[h['name']]} {h['currency']}"
     for h in hotels]
)
    prompt = f"""
You are an AI hotel booking assistant.

User request:
{user_query}

Hotels available:
{hotel_text}

Task:
- Recommend the best hotel(s) under user's budget (if specified)
- Only recommend hotels with rating >= 3
- Include the computed total fare
- Format answer as:

Welcome to AI Hotel booking system:

1. Hotel: {{hotel name}}
2. Fare should be: {{total fare}} for {{adults}} adults and {{children}} children
3. Sea front or not
4. Hotel Rating is : {{rating}}
5. Sight seeing places to roam around nearer to the hotel
"""
    # response = client.text_generation(prompt, max_new_tokens=500, temperature=0.3)
    # return response
    response = client.chat_completion(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": "You are an AI hotel booking assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.3)

    # Extract text
    return response.choices[0].message["content"]

# -------------------
# Streamlit UI
# -------------------
def main():
    st.title("🏨 AI Hotel Booking with Llama-3.3-70B-Instruct")

    user_input = st.text_area("Enter your hotel booking request:", 
        "Welcome to AI Hotel booking system. How may I help you?")

    if st.button("Find Hotels"):
        prefs = extract_preferences(user_input)

        candidate_hotels = [h for h in HOTELS if prefs["budget"] is None or (h["price_adult"] <= prefs["budget"] and h["price_child"] <= prefs["budget"])]
        fares = {}

        if prefs["check_in"] and prefs["check_out"]:
            for hotel in candidate_hotels:
                st.subheader(f"📊 Day-wise pricing for {hotel['name']}")
                df, total_fare = daywise_breakdown(hotel, prefs["check_in"], prefs["check_out"], prefs["adults"], prefs["children"])
                fares[hotel["name"]] = total_fare
                st.table(df)
                st.write(f"💰 Total fare for stay: {total_fare} {hotel['currency']}")

        st.subheader("🤖 AI Recommendation")
        answer = llama_recommendation(user_input, candidate_hotels, fares, prefs)
        st.write(answer)

if __name__ == "__main__":
    main()
