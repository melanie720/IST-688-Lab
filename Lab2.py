import pymupdf
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError, NotFoundError

# Show title and description.
st.title("📄 Mel's Summarizer App")
st.caption("• Upload a text or pdf file and get a summary.")
st.caption("• Choose language and summary type to the left if you like.. or not.")

# Retrieving API key.
openai_api_key = st.secrets.OPENAI_API_KEY

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Checking if API key is valid.
try:
    client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

# Creating a function to read PDF files.
def read_pdf(pdf_file):
    # 1. Read the uploaded file into memory as bytes
    file_bytes = pdf_file.getvalue()

    # Collect each page's text here so it can be joined at the end
    full_text = []

    # 2. Open the PDF from memory using the stream argument
    with pymupdf.open(stream = file_bytes, filetype = "pdf") as doc:
        st.success(f"Successfully loaded: {pdf_file.name}")

        # 3. Iterate through pages and extract text
        for page_num, page in enumerate(doc):
            # Extract text from the current page
            page_text = page.get_text()
            full_text.append(page_text)

    return "\n".join(full_text)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader("Document (.txt or .pdf)", type=("txt", "pdf"))

# Added a button:
if st.button("Submit", type="primary", disabled=not uploaded_file):

    # Process the uploaded file.
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'txt':
        document = uploaded_file.getvalue().decode()
    elif file_extension == 'pdf':
        document = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document}. Provide a summary in language {st.session_state.language_select or 'English'} and make sure it is {st.session_state.summary_type_select or '100 words'}.",
        }
    ]

    # Setting model.
    model_id = st.session_state.model

    # Need a spinner:
    with st.spinner(f"Reading your document with {model_id}...", show_time=True):
        # Handling model problems separately from key problems.
        try:
            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)

        # A 404 means the model is retired, renamed, or not on this account.
        except NotFoundError as e:
            st.error(
                f"**{model_id}** isn't available. "
                "It may have been deprecated, or your account may not have access. "
                "Pick a different model."
            )
            # Show OpenAI's error message.
            st.caption(f"OpenAI said: {e}")

        # Anything other errors from the API
        except OpenAIError as e:
            st.error(f"The request failed ({type(e).__name__}): {e}")