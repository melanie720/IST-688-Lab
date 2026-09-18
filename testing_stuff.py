import os
import streamlit as st
import pymupdf as fitz

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, "Lab-04-Data")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = "".join([page.get_text() for page in doc])
    return text

for file in os.listdir(target_path):
    text = extract_text_from_pdf(target_path + '/' + file)
    print(text)


###########################################################################

#### QUERYING A COLLECTION — ONLY USED FOR TESTING ####
topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.client
    response = client.embeddings.create(
        input=topic,
        model='text-embedding-3-small'
    )

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # The number of closest documents to return
    )

    # Display the results
    st.subheader(f'Results for: {topic}')

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]

        st.write(f'**{i+1}. {doc_id}**')

else:
    st.info('Enter a topic in the sidebar to search the collection')