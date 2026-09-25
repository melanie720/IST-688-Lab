import streamlit as st

# Page Configurations
st.set_page_config(
    page_title='LabApp',
    page_icon=':material/science:',
    layout="centered",
    menu_items={
        'About': "This is my lab app!"
    }
)

page1 = st.Page('Lab1.py', title='Lab 1', icon=':material/description:')
page2 = st.Page('Lab2.py', title='Lab 2', icon=':material/description:')
page3 = st.Page('Lab3.py', title='Lab 3', icon=':material/description:')
page4 = st.Page('Lab4.py', title='Lab 4', icon=':material/description:')
page5 = st.Page('Lab5.py', title='Lab 5', icon=':material/description:', default=True)

pg = st.navigation([page1, page2, page3, page4, page5], position='top')

# CSS for Containers
css = '''
    <style>
        .st-key-options_container, .st-key-model_container {
            background-color: #ffffff;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        .st-key-options_container .stSelectbox > div > div,
        .st-key-model_container .stSelectbox > div > div {
            border: 2px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }

        .st-key-options_heading p,
        .st-key-model_heading p {
            font-size: 16px;
            font-weight: 600;
            line-height: 1.4;
            color: var(--text-color);
            margin-bottom: 0;
        }

        .st-key-options_heading [data-testid="stIconMaterial"],
        .st-key-model_heading [data-testid="stIconMaterial"] {
            font-size: 18px;
            vertical-align: -3px;
        }

        .st-key-options_heading,
        .st-key-model_heading {
            margin-bottom: -0.5rem;
        }
    </style>
'''

st.html(css)

# Sidebar Title
st.sidebar.header(":material/settings: Summary Options")

# Container 1: Output Settings
with st.sidebar.container(border=True, height="content", key="options_container"):
    st.write("")

    with st.container(key="options_heading"):
        st.write(":material/tune: Output settings:")

    st.write("")

    st.session_state.language_select = st.selectbox(
        ':material/translate: Select language:',
        ('English', 'Spanish', 'French', 'Mandarin'),
        index=None
    )

    st.write("")

    st.session_state.summary_type_select = st.selectbox(
        ':material/summarize: Select type of summary:',
        ('100 words', '2 connecting paragraphs', '5 bullet points'),
        index=None
    )

    st.write("")

# Container 2: Model Selection
with st.sidebar.container(border=True, height="content", key="model_container"):
    st.write("")

    with st.container(key="model_heading"):
        st.write(":material/computer: Model selection:")

    st.write("")

    st.session_state.advanced = st.checkbox('Use Advanced Model')

    st.write("")

    # Setting LLM model based on user selection:
    if not st.session_state.advanced:
        st.markdown(
            '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using OpenAI\'s GPT-5.4 Nano</p>',
            unsafe_allow_html=True,
        )
        st.caption('_• Fast, concise, low-cost_')
        st.session_state.model = "gpt-5.4-nano"
    else:
        st.markdown(
            '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using OpenAI\'s GPT-5.6 Terra</p>',
            unsafe_allow_html=True,
        )
        st.caption('_• Detailed, thorough, a little slower_')
        st.session_state.model = "gpt-5.6-terra"

    st.write("")

pg.run()