# import openai
import psycopg2
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import openai
from openai import OpenAI

from google.generativeai import GenerationConfig


load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gemini_api_key)


conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port = os.getenv("PORT"),
    user = os.getenv("USER")

)

cursor = conn.cursor()

documents  =[
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "It encompasses fields like machine learning, natural language processing, and computer vision.",
    "AI systems learn and adapt from data to perform tasks with minimal human intervention.",
    "Applications of AI include healthcare, finance, robotics, and autonomous vehicles.",
    "Ethical considerations in AI involve bias, privacy, and decision-making transparency."
]

st.title("Q&A System about Artificial Intelligence")
st.write("This application demonstrates how the RAG system is working while question answering ")

# create embeddings and displaying it
st.header("These are the documents:")

for i , doc in enumerate(documents , start=1):
    st.write(f"{i}.{doc}")

embeddings = []

st.header("These are the embeddings:")

for doc in documents:
    # embedding_res = openai.Embedding.create(model="text-embedding-ada-002" , input=doc)
    embedding_res = genai.embed_content(
    model="models/text-embedding-004", content=doc, output_dimensionality=768   #max dimensionality for gemini openai has 1536 gemini is better 
)
    embedding = embedding_res['embedding']
    embeddings.append(embedding)

    cursor.execute(
        "INSERT INTO document_chunks (doc , embedding) VALUES (%s , %s)" , 
        (doc , embedding)
    )

    st.write(f"**embedding of chunk** '{doc}':")
    st.write(embedding)

conn.commit()

# Retrieving Relevant Chunks for a Question

st.header("Retrieve Relevant Chunks")
question = st.text_input("Enter Your Question:")

def get_relevant_chunks(question , top_k=3):
    question_embedding_res = genai.embed_content(
       model="models/text-embedding-004", content=question, output_dimensionality=768 
    )
    question_embedding = question_embedding_res['embedding']


    # Query top-k most relevant embeddings

    cursor.execute("""
      SELECT doc
      FROM document_chunks
      ORDER BY embedding <-> %s::vector
      LIMIT %s
  """, (question_embedding, top_k))

    relevant_chunks = [row[0] for row in cursor.fetchall()]

    return relevant_chunks


if question:
    st.write(f"**Embedding for **'{question}'")
    question_embedding_res = genai.embed_content(
       model="models/text-embedding-004", content=question, output_dimensionality=768 
    )
    question_embedding = question_embedding_res['embedding']

    st.write(question_embedding)

    relevant_chunks = get_relevant_chunks(question)

    st.write("Top relevant chunks retrieved:")

    for i , chunk in enumerate(relevant_chunks , start=1):
        st.write(f'{i}. {chunk}')


    # Generating Answer

    st.header("Generating Answer for Using Gemini:")

    context = "\n".join(f"{i+1}. {chunk}" for i , chunk in enumerate(relevant_chunks))
    prompt = f"Using following information:\n{context}\n Give answer to the following {question}"
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

   
# using normal gemini command bu gemini is compatible with OPEN AI LIBRARIES
    response = model.generate_content(prompt)
    answer = response.text
    st.write("**Generated Answer**:")
    st.write(answer)
# Generation using open AI

# Note : We are not using GPT Model so the answer maybe same we are just using gemini using openai library
    st.header("Generating Answer for Using OpenAI:")
    st.write("**Note : We are not using GPT Model so the answer maybe same we are just using gemini using openai library**")
    client = OpenAI(
        api_key=os.getenv("GEMINI_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    st.write("**Generated Answer**:")
    # st.write(answer)
    st.write(response.choices[0].message.content)


cursor.close()
conn.close()