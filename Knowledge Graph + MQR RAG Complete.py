import os
import time
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
from langchain_openai import AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
start_time = time.time() #calculate how long it takes for the program to run
load_dotenv()

key = openai.api_key
key = os.environ["OPENAI_API_KEY"]
database_pass = 'iloveifrs9somuch'

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = 'iloveifrs9somuch'
driver = GraphDatabase.driver(uri, auth=(username, password))

# We now need to create an a vector index in Neo4j. This is essentially like a vector database.
# Once created we can run the following code to access it. I have named mine index_1.
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url="bolt://localhost:7687",
    username="neo4j",
    password='iloveifrs9somuch',
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

def generate_response1(question, vector_index, API_key):
    # Initialize the AzureChatOpenAI instance
    
    #Allows for you to choose between 3.5 and azure 4o
    if API_key == 0:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-gs-v1",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            verbose=False,
            temperature=0,
        )
    else:
        try:
            llm = ChatOpenAI(api_key=API_key, model="gpt-3.5-turbo", temperature=0) 
        except:
            print("API key incorrect")
    
 
    # Create a prompt template
    template = """
    You are a regulatory assistant, answer the question in based on the context provided. If you can't answer
    the question, use your own knowledge base but please distinguish between the information you get from the context provided and 
    the information in your own knowledge base.
        
    Context: {context}

    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # Define the context generator function
    def context_generator(question):
        MQR = True
        list_of_keys = []
        context_document = []        
        if MQR == True:
            llm = ChatOpenAI(temperature=0)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=vector_index.as_retriever(), llm=llm

            ) #Create the multiquery retriever            
            unique_docs = retriever_from_llm.invoke(question) #create multiple questions (3) and then find their corresponding keys/ meta data
            
            print("number of keys:",len(unique_docs)) # The number of keys we have
            for i in range (0,len(unique_docs)):
                print(unique_docs[i].page_content)
                list_of_keys.append(unique_docs[i].page_content[6:]) #adds the keys to the list and removes 'key: '
            #print(list_of_keys)

        else:
            response_key = vector_index.similarity_search(question) #Uses similarity search to return the nearest embedding's Neo4j key. Returns 4 keys by default.
            for i in range(0, len(response_key)):
                new_response_key = response_key[i].page_content
                list_of_keys.append(new_response_key[6:]) #removes the word 'key:  ' from the start of each key.
            #print(list_of_keys)
        
        for i in range(0, len(list_of_keys)):
            initial_node_id = list_of_keys[i]
            connected_nodes = get_connected_nodes(initial_node_id) #Retrieves the text from the Section linked to each chunk. (This provides full context for that chunk)
            
            to_be_added = Document(page_content =generate_context(connected_nodes)) #Ensures that each item added is unique
            if to_be_added not in context_document:
                context_document.append(to_be_added)
                print("document added")
            else:
                print("There is a repeat.")
        #print(context_document)
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
import streamlit as st
st.markdown("<div class = 'sticky-header'><h3>Reggie - 4most's Regulatory Assistant</h3></div>", unsafe_allow_html=True)
#st.header("Reggie - 4most's Regulatory Assistant")
st.logo("4most.png")

# Apply the custom CSS
st.markdown("""
    <style>
    .sticky-header {
        position: fixed;
        background-color: white;
        top: 50px; /* Adjust the header to be flush with the top of the screen */
        width: 100%;
        margin: 0;
        padding: 0;
        text-align: left;
        z-index: 9999; /* Ensure the header stays on top */
        color: #272640; /* Text color */
    }
            
    /* Change the background color of the sidebar */
    [data-testid="stSidebar"] {
        background-color: #272640;
    }
    
    /* Change the text color in the sidebar */
    [data-testid="stSidebar"] * {
        TextColor: #31BCC3;
    }
            
    /* Style the slider labels */
    [data-testid="stSlider"] div[data-testid="stMarkdownContainer"] {
        color: #31BCC3; /* Change the color of the labels */
    }        
    
    [data-testid="stSlider"] input[type="range"]::-ms-thumb {
    color: #31BCC3; /* Change the color of the slider thumb for IE and Edge */
    }

    /* Specifically target the text inside the selectbox to be black */
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stMarkdownContainer"] {
        color: #31BCC3 !important;
    }
    /* Change the color of the 'Help' link */
    [data-testid="stSidebar"] a {
        color: #31BCC3; /* Example: a custom orange color */
    }        

    /* Ensure the dropdown items themselves also appear in black */
    [data-testid="stSidebar"] .stSelectbox div[role="listbox"] span {
        color: white !important;
    }
    /* Add a hover effect */
    [data-testid="stSidebar"] a:hover {
        color: #FFD700; /* Example: a gold color on hover */
        text-decoration: underline; /* Add underline on hover */
    }
            
    /* Style the labels of text input boxes */
    [data-testid="stTextInput"] label {
        color: #31BCC3; /* Change the color of the label text */
    }
    """,
    unsafe_allow_html=True
)

with st.form("my form", clear_on_submit=False):
    add_selectbox = st.sidebar.selectbox(
        "Which LLM would you like to use?",
        ("GPT-4o", "GPT-3.5-turbo")
    )

    # Using "with" notation
    with st.sidebar:
        API_key = 0
        if add_selectbox == 'GPT-3.5-turbo':
            API_key = st.text_input("API key")
        
        type_speed = st.slider("Typewritter effect speed",0,10,1)
        #Number_of_chunks = st.slider("Speed vs Accuracy",1,4,1)
        st.form_submit_button(label="Submit")
        st.markdown('[Help](https://chatgpt.com/)', unsafe_allow_html=True)
        #st.logo("./Other Random/4most_logo-1.png")
    print(add_selectbox)
    print(API_key)

with st.chat_message(name="assistant",avatar="😀"): #you can change the avatar to an image as well
    st.write("Hi! How can I help you today?")

    if add_selectbox == 'GPT-3.5-turbo' :
        #st.write(API_key)
        if API_key == 0:
                response = "You have chosen GPT-3.5-turbo, please insert your API key"
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
            time.sleep(type_speed/100)

    with st.chat_message(name="assistant",avatar="😀"):
        with st.spinner("Thinking..."):
            response = generate_response1(prompt,vector_index, API_key)
            st.write_stream(stream_data(response))
    st.session_state.messages.append({"role": "assistant", "content": response})


# Don't forget to close the driver connection when done
driver.close()
print("--- %s seconds ---" % (time.time() - start_time))