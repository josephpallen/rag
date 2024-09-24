import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
### GPT 3.5 turbo key

## Imports
import os
import openai
import dotenv
import copy
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


## Downloads
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


## Code start
#"""Initialise Azure"""

def initialise():
    dotenv.load_dotenv()
    use_azure_active_directory = False  # Set this flag to True if you are using Azure Active Directory
    if not use_azure_active_directory:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2023-09-01-preview"
        )
        return client
initialise()

#"""Load docs from PDF"""
@st.cache_data
def load_docs():
    def load_docs_2(file_path):
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        #print("File has been processed successfully")
        return docs

    def read_pdfs_from_folder(folder_path):
        pdf_texts=[]
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path,filename)
                text = load_docs_2(file_path) #load_docs is the same function from the above text other than the file directory has been changed
                pdf_texts.append(text)
                print(filename,"has been loaded and processed")
            else:
                ("File type not compatible. Please use a PDF document")
        return pdf_texts
    folder_path = r"C:\Users\JosephAllen\ragmodel0.1\Scripts\Vector_docs"
    compiled_text=read_pdfs_from_folder(folder_path) # Should take around 37 seconds
    return compiled_text
compiled_text = load_docs()

#"""Lemmatisation"""
@st.cache_data
def lemmatisation(_compiled_text):
    lemmatizer = WordNetLemmatizer()

    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemma_creater(sentence):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]
        
        lemmatized_sentence = [
            lemmatizer.lemmatize(word, tag) if tag else word
            for word, tag in wordnet_tagged
        ]
        
        return " ".join(lemmatized_sentence)


    def lemma_applier_all(documents):
        processed_documents = [copy.deepcopy(doc_group) for doc_group in documents]
        for document_group in processed_documents:
            for doc in document_group:
                doc.page_content = lemma_creater(doc.page_content)
        print("All",len(documents),"documents have been lemmatised")
        return processed_documents

    # Create shallow copies of the original documents
    compiled_text_copy = copy.deepcopy(compiled_text)
    # Process the documents
    output_all = lemma_applier_all(compiled_text_copy)
    # Lemmatised documents
    lemma_document_all = output_all
    # Now you can still access the original documents
    original_compiled_text = compiled_text
    return lemma_document_all, original_compiled_text
lemma_document_all, original_compiled_text = lemmatisation(compiled_text)

#"""Stemming"""
@st.cache_data
def stemmisation(_lemma_document_all):
    stemmer = PorterStemmer()
    
    # Ensure consistency with variable names
    # creates a new object and recursively inserts copies into it of the objects found in the original.
    place_holder_all = copy.deepcopy(compiled_text)
    
    words_to_remove = {    
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    } #Note that all words must be in lower case
    
    def preprocess_text(text):
        #Helper function to preprocess text: lowercasing and removing punctuation.
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def remove_words_from_content_all(documents, words_to_remove, apply_stemming=False):
        for doc_list in documents:
            for doc in doc_list:
                # Preprocess text
                processed_text = preprocess_text(doc.page_content)
                # Split the content into words
                words = word_tokenize(processed_text)
                # Filter out the words that should be removed
                filtered_words_all = [word for word in words if word not in words_to_remove]
                # Apply stemming if required
                if apply_stemming:
                    filtered_words_all = [stemmer.stem(word) for word in filtered_words_all]
                # Join the filtered words back into a single string
                doc.page_content = ' '.join(filtered_words_all)
        return documents
    
    # Remove specified words from each *Lemmatised* document's page_content, and the normalise by removing punctuation etc.

    normalised_doc_all= remove_words_from_content_all(lemma_document_all, words_to_remove, apply_stemming=False) #select false if lemmatised already
    
    # Print updated documents to verify
    #print("unfiltered:", original_compiled_text[1])
    #print("filtered:", normalised_doc_all[1])
    return normalised_doc_all
normalised_doc_all = stemmisation(lemma_document_all)

#"""Chunk the Data functions"""
### I decided to use only recursive or token

def recursive_chunk_all (documents):
    #print(len(documents))
    splitted = []
    for i in range(0,len(documents)):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents[i])
        #print(splits)
        splitted.append(splits)
    #print(len(splitted))
    return splitted

def token_chunk_all (documents):
    splitted = []
    for i in range(0,len(documents)):
        text_splitter = TokenTextSplitter()
        splits = text_splitter.split_documents(documents[i])
        #print(splits)
        splitted.append(splits)

    return splitted

#"""Create the vector store functions"""

client = openai.AzureOpenAI(
     azure_endpoint=os.getenv("AZURE_EMBED_ENDPOINT"),
     api_key=os.getenv("OPENAI_API_KEY"),
     api_version="2023-05-15"
)

embedding_function =  AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002-wus",
    openai_api_version="2023-05-15",
)

gpt_model = "3.5"
def embedder(chunk, preference):
    if preference == "4o":
        db = FAISS.from_documents(chunk[0], embedding_function) #gpt4o
        for sublist in chunk[1:]:
            db.add_documents(documents=sublist)
        return db
    else:
        db = FAISS.from_documents(chunk[0], embedding=OpenAIEmbeddings(api_key=key))  #gpt3.5 turbo
        for sublist in chunk[1:]:
            db.add_documents(documents=sublist)
        print("Chunks have been embedded")
        return db

#"""**********************************************************"""
#"""
#Please choose between the normalised (True) or original document (False)
#"""
normalised_document_choice = False
#"""
#Please choose between recursive chunking (True) or Token chunking (False)
#"""
chunking_method = True
#"""
#Please enter the GPT model you wish to use for the embedding (4o or 3.5 turbo)
#"""
chunking_model = "3.5"

#"""**********************************************************"""

if normalised_document_choice == True:
    doc_choice = normalised_doc_all
else:
    doc_choice = original_compiled_text

if chunking_method == True:
    chunked_document = recursive_chunk_all(doc_choice)
else:
    chunked_document = token_chunk_all(doc_choice)

#create the vector store
@st.cache_resource
def vectorstore_creation(_chunked_document, _chunking_model):
    vectorstore = embedder(chunked_document,chunking_model)
    return vectorstore
vectorstore = vectorstore_creation(chunked_document, chunking_model)

from langchain.chains import LLMChain
def context_generator(question, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    context = retriever.get_relevant_documents(question)
    return context

# Create the prompt template
template = """
You are a regulatory assistant, answer the question in based on the context provided. If you can't answer
the question, use your own knowledge base but please distinguish between the information you get from the context provided and 
the information in your own knowledge base. 

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Function to generate response
def generate_response1(question, vectorstore):
    # Initialize the AzureChatOpenAI instance
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-gs-v1",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        verbose=False,
        temperature=0,
    )
    # If you can't answer

    template = """
    You are a regulatory assistant, answer the question in based on the context provided. If you can't answer
    the question, use your own knowledge base but please distinguish between the information you get from the context provided and 
    the information in your own knowledge base.
        
    Context: {context}

    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # Define the context generator function
    def context_generator(question, vectorstore):
        retriever = vectorstore.as_retriever(search_type="similarity") #, search_kwargs={"k": 6}
        docs = retriever.invoke(question)
        return docs

    # Format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Generate context and format the prompt
    docs = context_generator(question, vectorstore)
    context_text = format_docs(docs)
    print(context_text)
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    
    # Create the RAG chain
    response = llm.invoke(formatted_prompt)
    print(response.content)
    
    return response.content

#""" ##################################################  
#We shall now implement the basic RAG model into streamlit"""

import time
import numpy as np
import pandas as pd
import streamlit as st
st.markdown("## Reggie - 4most's regulatory assistant")
#st.logo("4most_logo-1.png")

add_selectbox = st.sidebar.selectbox(
    "Which LLM would you like to use?",
    ("GPT-4o", "Llama3")
)

# Using "with" notation
with st.sidebar:
    API_key = 0
    if add_selectbox == 'Llama3':
        API_key = st.text_input("API key")
    st.page_link("https://google.co.uk",label="Help")
    
print(add_selectbox)
print(API_key)

with st.chat_message(name="assistant", avatar="😀"): #you can change the avatar to an image as well
    st.write("Hi! I'm Reggie, your regulatory chatbot. How can I help you to day?")

    if add_selectbox == 'Llama3' :
        #st.write(API_key)
        if API_key == 0 or ' ':
                response = "You have chosen Llama, please insert your API key"
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Please enter your questions here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
        
    def stream_data(output): #make the output look nicer
        for word in output.split(" "):
            yield word + " "
            time.sleep(0.05)

    with st.chat_message("assistant", avatar="😀"):
        with st.spinner("Thinking..."):
            start_time = time.time()  # Start the timer
            response = generate_response1(prompt,vectorstore)
            st.write_stream(stream_data(response))
            end_time = time.time()  # End the timer
            
            response_time = end_time - start_time
            st.write(f"Response Time: {response_time:.2f} seconds")
    st.session_state.messages.append({"role": "assistant", "content": response})
