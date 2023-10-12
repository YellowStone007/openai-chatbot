User
import streamlit as st
from langchain embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def load_document(file):
	import os
	name, extension = os.path.splitext(file)

	if extension == '.pdf':
		from langchain.document_loaders import PyPDFLoader
		print(f'Loading {file}')
		loader = PyPDFLoader(file)
	elif extension == '.docx':
		from langchain.document_loaders import Docx2txtLoader
		print(f'Loading {file}')
		loader = Docx2txtLoader(file)
	elif extension == '.txt':
		from langchain.document_loaders import TextLoader
		loader = TextLoader(file)
	else:
		print('Document format is not supported!')
		return None


	data = loader.load()
	return data

#wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
	from langchain.document_loaders import WikipediaLoader
	loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs) 
	data = loader.load()
	return data

 
def chunk_data(data, chunk_size=256, chunk_overlap=20):
	from langchain.text_splitter import RecursiveCharacterTextSplitter
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	chunks = text_splitter.split_documents(data)
	return chunks



def create_embeddings(chunks):
	embeddings = OpenAIEmbeddings()
	vector_store = Chroma.from_documents(chunks, embeddings)
	return vector_store


def ask_and_get_answer(vector_store, q, k=3):
	from langchain.chains import RetrievalQA
	from langchain.chat_models import ChatOpenAI
	
	llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
	
	retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
	
	chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

	answer = chain.run(q)
	return answer


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
   # print(f'Total Tokens: {total_tokens}')
   # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
   return total_tokens, Total_tokens / 1000 * 0.0004



#Creating the sidebar

if __name__ == "__main__":
	import os
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(), override=True)

	st.image('img.png')
	st.subheader('👩‍🚀🚀 Mariia/Yellow.Stone Chatbot')
	with st.sidebar:
		api_key = st.text_input('OpenAI API Key:', type='password')
		if api_key:
			os.environ['OPENAI_API_KEY'] = api_key

		uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
		chunk_size = st.number_input('k', main_value=1, max_value=20, value=3)
		add_data = st.button('Add Data')
