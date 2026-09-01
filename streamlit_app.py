import pymupdf
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError, NotFoundError

page1 = st.Page('Lab1.py', title = 'Lab1', icon = ':material/coffee:')
page2 = st.Page('Lab2.py', title = 'Lab2', icon = ':material/coffee:')

pg = st.navigation([page1, page2])
st.set_page_config(page_title = 'LabApp', page_icon = ':material/coffee:')
pg.run()

language_select = st.sidebar.selectbox(
    'Select language:',
    ('English', 'Spanish', 'French', 'Mandarin')
)

summary_type_select = st.sidebar.selectbox(
    'Select type of summary:',
    ('100 words', '2 connecting paragraphs', '5 bullet points')
)

if 'language_select' not in st.session_state:
    st.session_state['language_select'] = language_select

if 'summary_type_select' not in st.session_state:
    st.session_state['summary_type_select'] = summary_type_select

st.write(st.session_state.language_select}"
st.write(st.session_state.summary_type_select}"