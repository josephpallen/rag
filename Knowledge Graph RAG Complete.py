import os
import openai
import dotenv
import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
openai.api_key = key
database_pass = 'iloveifrs9somuch'

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "iloveifrs9somuch"
driver = GraphDatabase.driver(uri, auth=(username, password))

# We now need to create an a vector index in Neo4j. This is essentially like a vector database.
# Once created we can run the following code to access it. I have named mine index_1.
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url="bolt://localhost:7687",
    username="neo4j",
    password=database_pass,
    database="neo4j",
    index_name='index_1',
    node_label="Chunk",
    text_node_properties=['key'],
    embedding_node_property='embedding',
)

# Define the function to find connected nodes
@st.cache_resource
def find_connected_nodes(_tx, initial_node_id):
    query = (
        """
        MATCH (n)-[r]->(m)
        WHERE n.key = $initial_node_id
        RETURN n, r, m
        """
    ) # This runs a cypher query that allows for us to retrieved all connected nodes to our 'initial node'
    result = _tx.run(query, initial_node_id=initial_node_id)
    return [(record["n"], record["r"], record["m"]) for record in result] #returns relevant information

@st.cache_resource
def get_connected_nodes(initial_node_id):
    with driver.session() as session:
        result = session.execute_read(find_connected_nodes, initial_node_id)
        return result

#Fetch context from connected node
#@st.cache_resource
def generate_context(_connected_nodes):
    for n, r, m in _connected_nodes:
        if m['text'] != 'None' and r.type == 'UNDER_SECTION':
            #print("Context found:",m['text'])
            return m['text']

# Function to generate response
def generate_response1(question, vector_index):
    # Initialize the AzureChatOpenAI instance
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-gs-v1",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        verbose=False,
        temperature=0,
    )
 
    # Create a prompt template
    template = """
    You are a regulatory assistant called Reggie, answer the question based on the documents provided. If you can't answer
    the question, reply "This question is outside my dataset. Please try again or see 'help' for advice on how to structure a question efficiently.".
        
    Context: {context}

    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # Define the context generator function
    #@st.cache_data
    def context_generator(question):
        response_key = vector_index.similarity_search(question, k =4) #Uses similarity search to return the nearest embedding's Neo4j key. Returns 4 keys by default., k=num_chunks
        list_of_keys = []
        context_document = []
        for i in range(0, len(response_key)):
            new_response_key = response_key[i].page_content
            list_of_keys.append(new_response_key[6:]) #removes the word 'key:  ' from the start of each key.
        #print(list_of_keys)

        for i in range(0, len(list_of_keys)):
            initial_node_id = list_of_keys[i]
            connected_nodes = get_connected_nodes(initial_node_id) #Retrieves the text from the Section linked to each chunk. (This provides full context for that chunk)
            context_document.append(Document(page_content =generate_context(connected_nodes) )) # Then adds all of the context into a list ## need to add meta data as well??
        #Need to ensure each document is unique:

        print(context_document[0])
        return context_document

    context_document = context_generator(question)
    
    # Format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Generate context and format the prompt
    context_text = format_docs(context_document)

    formatted_prompt = prompt_template.format(context=context_text, question=question) # Adds context to prompt
    #print(formatted_prompt)
    
    # Create the RAG response
    response = llm.invoke(formatted_prompt) 
    print(response.content)
    
    return response.content
##################

import time
import numpy as np
import pandas as pd
import streamlit as st
st.markdown("## Reggie - 4most's regulatory assistant")
#st.logo("4most_logo-1.png")
with st.form("my form", clear_on_submit=False):
    add_selectbox = st.sidebar.selectbox(
        "Which LLM would you like to use?",
        ("GPT-4o", "Llama3")
    )

    # Using "with" notation
    with st.sidebar:
        API_key = 0
        if add_selectbox == 'Llama3':
            API_key = st.text_input("API key")
        
        type_speed = st.slider("Typewritter effect speed",0,10,1)
        #Number_of_chunks = st.slider("Speed vs Accuracy",1,4,1)
        st.form_submit_button(label="Submit")
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
            time.sleep(type_speed/25)

    with st.chat_message("assistant", avatar="😀"):
        with st.spinner("Thinking..."):
            start_time = time.time()  # Start the timer
            response = generate_response1(prompt, vector_index)
            st.write_stream(stream_data(response))
            end_time = time.time()  # End the timer
            
            response_time = end_time - start_time
            st.write(f"Response Time: {response_time:.2f} seconds")
    st.session_state.messages.append({"role": "assistant", "content": response})


# Don't forget to close the driver connection when done
driver.close()