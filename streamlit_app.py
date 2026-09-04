import streamlit as st

page1 = st.Page('Lab1.py', title='Lab1', icon=':material/description:')
page2 = st.Page('Lab2.py', title='Lab2', icon=':material/description:', default=True)

st.set_page_config(page_title='LabApp', page_icon=':material/science:')
pg = st.navigation({'Assignments': [page1, page2]}, position='top')

st.sidebar.markdown("<h3 style='color:#8B0000; '>Summary Options</h3>", unsafe_allow_html=True)

st.sidebar.subheader(':material/translate: Language')
st.session_state['language_select'] = st.sidebar.selectbox(
    'Select language:',
    ('English', 'Spanish', 'French', 'Mandarin'),
    index = None
)

st.sidebar.subheader(':material/summarize: Type')
st.session_state['summary_type_select'] = st.sidebar.selectbox(
    'Select type of summary:',
    ('100 words', '2 connecting paragraphs', '5 bullet points'),
    index = None
)

st.sidebar.divider()

st.sidebar.subheader(':material/computer: Model Selection')
st.session_state['advanced'] = st.sidebar.checkbox('Use Advanced Model')

st.sidebar.write('')

if not st.session_state.advanced:
    st.sidebar.write('**Currently using GPT-5.4 Nano**')
    st.sidebar.caption('• Fast, concise, low-cost')
    st.session_state['model'] = "gpt-5.4-nano"
else:
    st.sidebar.write('**Now using GPT-5.6 Terra**')
    st.sidebar.caption('• Detailed, thorough, a little slower')
    st.session_state['model'] = "gpt-5.6-terra"

pg.run()