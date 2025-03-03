from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import os
from llama_index.core import (VectorStoreIndex,SimpleDirectoryReader,load_index_from_storage,StorageContext)
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

#Setting custom llm and embeddings
Settings.llm = Groq(
    model="llama3-70b-8192",
    temperature=0.3,
    max_tokens=3000
)

#Setting custom embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-small-en-v1.5"
)


def get_query_engine():
    PER_DIR ="./storage"

    if not os.path.exists(PER_DIR):
       #Loading and storing documents int vector database
       documents = SimpleDirectoryReader("data").load_data()

       #Storing data into vector db
       index = VectorStoreIndex.from_documents(documents=documents)

       #Storing data
       index.storage_context.persist(persist_dir=PER_DIR)

    else:
        #If already db exits
        storage_context = StorageContext.from_defaults(persist_dir=PER_DIR)

        #Getting index database
        index = load_index_from_storage(storage_context=storage_context)


    #Getting query_engine
    query_engine = index.as_query_engine()

    return query_engine


#Testing
# query_engine = get_query_engine()
# print(query_engine.query("what is constitution?"))


#Creating UI
if __name__ =="__main__":

    st.title("Indian Constitution Q&A")
    query = st.text_input("Enter Your Question : ")
    submit = st.button("Process")
    try:
        #Getting a query_engine
        query_engine = get_query_engine()

        if submit or query:
            if query:
                response = query_engine.query(query)
                st.header("**Answer**")

                #Answer
                st.write(response.response)

                #Getting source document
                source_doc = response.source_nodes

                #Providing source
                st.download_button("Download Source",source_doc[0].text,file_name="question_source")

    except Exception as e:
        st.write("Something went wrong ",e)

