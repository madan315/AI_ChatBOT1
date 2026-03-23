import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Chatbot 🤖")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if user_input:
    vec = vectorizer.transform([user_input])
    response = model.predict(vec)[0]
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

for role, msg in st.session_state.history:
    st.write(f"**{role}:** {msg}")
