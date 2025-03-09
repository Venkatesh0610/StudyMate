import streamlit as st
import os
import warnings
import time
from PyPDF2 import PdfReader

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Hugging Face API Token (Replace with your token)
HF_TOKEN = "************************************"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Set environment variables for authentication  
os.environ["GOOGLE_API_KEY"] = "************************************" # Set API key for Google's generative model authentication


# Ignore warnings
warnings.filterwarnings("ignore")

# Streamlit Page Setup
st.set_page_config(page_title="StudyMate", layout="wide")

# Sidebar for Data Source Selection
st.sidebar.header("Select Data Source")
data_source = st.sidebar.radio("Choose Input Type", ["URL", "PDF"])
print("Data source : ", data_source)

# Initialize variables
docs = []
url_list = []
retriever = None

if data_source == "URL":
    urls = st.sidebar.text_area("Enter URLs (one per line)", "")
    
    if urls:
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        st.sidebar.write("### Saved URLs:")
        for url in url_list:
            st.sidebar.markdown(f"- [{url}]({url})")
        
        try:
            # Attempt to load the URLs
            loader = WebBaseLoader(url_list)
            docs = loader.load()
            if not docs:  # Check if no content was loaded
                st.sidebar.error("No content could be extracted. Please check the URLs.")
        
        except Exception as e:
            st.sidebar.error("Invalid URL detected. Please check the URLs and try again.")
            print(f"Error loading URLs: {e}")  # Print error for debugging

elif data_source == "PDF":
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            docs.append(text)

# Process Documents
if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    # Embeddings & Vector Store
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="sentence-transformers/all-mpnet-base-v2")
    if data_source == "URL":        
        chunks = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(chunks, embeddings)
    elif data_source == "PDF":
        chunks = text_splitter.split_text("\n".join(docs))
        vectorstore = Chroma.from_texts(chunks, embeddings)
    else:
        chunks = []
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("Number of Documents in Vector Store:", len(vectorstore.get()["documents"]))
    print("Retrieved Docs:", retriever)  # Check first 2 docs

# LLM Model
# model = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-alpha",
#     model_kwargs={"temperature": 0.5, "max_new_tokens": 512, "max_length": 64}
# )

# Set up the underlying language model (LLM) with the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")

# Chatbot UI (Main Component)
st.header("StudyMate : AI Chatbot for Student Learning📚")
st.write("AI-powered chatbot that generates structured study notes from URLs or PDFs, providing definitions, key points, examples, and real-world applications to enhance learning. 🚀")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input in the main area
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to chat history and display immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if retriever:
        with st.spinner("Generating response..."):
            time.sleep(1)  # Simulate processing delay

            # Retrieve relevant documents
            docs_rel = retriever.get_relevant_documents(user_input)
            print(docs_rel)
            # Combine retrieved documents into context
            context = "\n\n".join([doc.page_content for doc in docs_rel])

            # Construct prompt
            prompt = f"""
            <|system|>>
            You are **StudyMate**, an AI Study Assistant designed to help college students by generating **comprehensive study notes** on any given topic.  
            Your response must be **clear, structured, and informative**, covering the following sections (If a section is not applicable, skip it but always provide a definition):

            1. **Definition:** Provide a precise definition of the topic.    
            2. **Key Points:** Summarize the most important aspects students should remember.  
            3. **Examples:** Offer real-world or practical examples for better understanding.  
            4. **Use Cases & Applications (if applicable):** Explain how this concept is applied in real life or specific fields.  

            For general queries, kindly introduce yourself as **StudyMate** and let users know:  
            *"To start using StudyMate, please add a URL or upload a PDF with study materials, and I will assist you in learning!"*  

            If the topic is **out of scope or not present in the provided data**, respond with:  
            *"I don't know" or "The requested information is not available in the provided content."*  
            
             Based on the following extracted study material, answer the user's question:

            ---  
            {context}  
            ---  

            Your goal is to be a **helpful, concise, and reliable study companion** for students.  
            </s>  
            <|user|>  
            {user_input}  
            </s>  
            <|assistant|>  
            """


            # Create an LLMChain with a prompt template
            prompt_template = PromptTemplate(template=prompt, input_variables=["question"])
            chain = LLMChain(llm=model, prompt=prompt_template)

            # Generate response
            bot_response = chain.run(question=user_input)

            # Append bot response to chat history
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})

            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(bot_response)
    else:
        st.warning("Please provide data (URLs or PDFs) in the sidebar to enable chatbot retrieval.")