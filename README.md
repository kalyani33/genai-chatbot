# genai-chatbot


## Chatbot Architecture
![llm architecture](https://github.com/kalyani33/genai-chatbot/blob/master/Screenshot_2024-07-22-08-04-48-612_com.udemy.android.ufb.jpg)

## Key Terminolgies to Understand:
### LLM
LLMs are large deep-learning models pre-trained on large amounts of data that can generate responses to user queries
### Embeddings
information representation of semantic meaning of a text or objects like audio,video,images etc.that are to be consumed by machine learning models or seamantic search algorithms(llm).
### Vector Store
is a database to store the embeddings where semantic search happens.

## Python Libraries Used
- **Streamlit**: For building interactive web apps(UI/UX) quickly with only a few lines of python code
- **PyPDF2**: A pure-python PDF library capable of splitting, merging, cropping, and transforming PDF files.
- **langchain**: a Python framework designed to streamline AI application development, focusing on real-time data processing and integration with Large Language Models (LLMs)

## Source Code

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = "sk-Wr5VzIVOwRoIyzTkQTjiaLQ6lSc84" #Pass your key here

#Upload PDF files
st.header("My first Chatbot")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )
        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
```

## Final Thoughts

![thoughts](https://github.com/kalyani33/genai-chatbot/blob/master/Screenshot_2024-07-22-22-47-11-150_com.udemy.android.ufb.jpg)
