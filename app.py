import streamlit as st
from langchain.schema import Document
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
st.header('A RAG App')



groqapi=''

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
uploaded_file=r'C:\Users\ASHWINI\Downloads\Cheenai_LTT.pdf'

#Read PDF with PyPDF2
text=''
pdf_reader=PyPDF2.PdfReader(uploaded_file)
for page in pdf_reader.pages:
    text += page.extract_text()+ '\n'

#split txt into chunks(strings)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_chunk=splitter.split_text(text)

#convert each chunk to documents format
docs=[Document(page_content=chunk) for chunk in text_chunk]
st.subheader('Document splitted successfully')

#create embeddings model
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#create FAISS vector store for documents
vectordb=FAISS.from_documents(docs,embeddings)  

st.success("FAISS VectorStore created successfully")
retriver=vectordb.as_retriever()



from langchain.chat_models import init_chat_model
model=init_chat_model(model='gemma2-9b-it',model_provider='groq',api_key=groqapi)

from langchain.prompts  import PromptTemplate

template="""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not present, just say no. Do not try to make up an answer.

 Context:
 {context}

 Question:
 {question}

 Helpful Answer:
"""

rag_prompt=PromptTemplate(input_variables=['context','question'],template=template)


user_query=st.text_input(" ? Ask a question about the PDF")

if user_query:
    relevant_docs=retriver.invoke(user_query)

    final_prompt=rag_prompt.format(context=relevant_docs,question=user_query)

    with st.spinner(" Generating answer..."):
        response=model.invoke(final_prompt)

    st.write('### Answer')
    st.write(response.content)
