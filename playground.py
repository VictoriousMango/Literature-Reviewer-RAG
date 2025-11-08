from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import json

class LiteratureReviewer:
    def __init__(self):
        load_dotenv()
        # Create Vector Store if it does not exit
        if not os.path.exists('db/faiss_index'):
            self.CreateVectorStore()
        self.db = self.LoadDB()
        self.retriever = self.db.as_retriever(search_type='similarity', search_kwargs={'k':3})
        self.model = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model="llama-3.3-70b-versatile", max_tokens=175, temperature=0.5)

    def CreateVectorStore(self, pdf_path:str ='assets/test.pdf') -> None:
        '''
        Parameter: pdf_path
        Tasks:
            1. Load PDF Documnet
            2. Split the documnet using RecurssiveCharacterTextSPlitter
            3. Create EMbeddings
            4. Create a Vector Store using FAISS
            5. save the embeddings into Vector Store
        '''
        # Step 1
        pdf = PyMuPDFLoader(file_path=pdf_path)
        document = pdf.load()

        # Step 2
        obj_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        splits = obj_splitter.split_documents(document)

        # Step 3
        model_name = 'all-MiniLM-L6-v2'
        embed = SentenceTransformerEmbeddings(model_name=model_name)

        # Step 4
        db = FAISS.from_documents(splits, embed)

        # Step 5
        db.save_local('db/faiss_index')
    
    def LoadDB(self) -> FAISS:
        '''
        Task: Load the FAISS Vector Store
            1. Create Embeddings object
            2. Load the FAISS Vector Store
        '''
        # Step 1
        model_name = 'all-MiniLM-L6-v2'
        embed = SentenceTransformerEmbeddings(model_name=model_name)

        # Step 2
        db = FAISS.load_local('db/faiss_index', embed, allow_dangerous_deserialization=True)
        return db
    
    def GetResponse(self, query:str) -> str:
        '''
        Parameter: query, question to be answered
        Tasks:
            1. Retireve the relavant answers from the Vector Store
            2. Create prompt template
            3. Get response from Groq LLM and return the response
        '''
        docs = self.retriever.invoke(query)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following quesiton based on the context below \nquestion:{question}\ncontext:{context}\n
            Rules: 
                1. It should be precise and concise
                2. It should be structured as points, nested points if required
                3. If the answer is not found in the context, respond with "Answer not found in the documnet
                4. Use formal, simple and technical language
                5. Avoid answering in paragraphs
            """
        )
        chain = prompt | self.model
        response = chain.invoke({"question": query, "context": docs})
        return response.content

if __name__ == "__main__":
    obj = LiteratureReviewer()
    Questions = [
        "Create a summary of the document.",
        "What are the key findings discussed in the document?",
        "Are there any limitations mentioned in the document?"
    ]
    with open("Result.json", "w") as f:
        data = {
            "Topic": "Literature Reviewer RAG Results",
            "QnA" : []
        }
        for question in Questions:
            answer = obj.GetResponse(query=question)
            data["QnA"].append({
                "Question": question,
                "Answer": answer
            })
        f.write(json.dumps(data, indent=4))
    print("Results saved to Result.txt")