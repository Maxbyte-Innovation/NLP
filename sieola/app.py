import credentials
import url
from flask import Flask, render_template, request
from langchain.document_loaders import DirectoryLoader,  SeleniumURLLoader, YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import os
os.environ['OPENAI_API_KEY']=credentials.api_key

app = Flask(__name__)

# import mysql.connector

# # Establish a database connection
# connection = mysql.connector.connect(
#     host="127.0.0.1",
#     user="root",
#     password="sieomysql",
#     database="maxbyte"
# )
# # Create a cursor
# cursor = connection.cursor()
# # Execute SQL query
# cursor.execute("SELECT * FROM file")
# result = cursor.fetchall()
# data=pd.DataFrame(result, columns=['S.no','SupervisorComments','Reason','Comments','SolutionGiven','CheckDescription','MachineName','CheckType','DepartmentName'])
# # data.to_csv('datacsv.csv')
# file='datacsv.csv'
# # Close the cursor and connection
# cursor.close()
# connection.close()


#splitting docs into chunks
text_splitter1=CharacterTextSplitter(chunk_size=2500,chunk_overlap=600) #PDF
text_splitter2=CharacterTextSplitter(chunk_size=3900,chunk_overlap=500) #URL
text_splitter3=CharacterTextSplitter(chunk_size=1000,chunk_overlap=100) #VDO
text_splitter4=CharacterTextSplitter(chunk_size=3000,chunk_overlap=300) #CSV

# loading files
loader1= DirectoryLoader("./files/",glob='**/*.pdf')
# loader2 = SeleniumURLLoader(urls=url.urls)
# loader3 = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=M0tZO_WN-o8", add_video_info=True
# )
loader4= DirectoryLoader("./files/",glob='**/*.csv')


#loading documents
docs1=loader1.load()
# docs2=loader2.load()
# docs3=loader3.load()
docs4=loader4.load()



texts=text_splitter1.split_documents(docs1)
# texts+=text_splitter2.split_documents(docs2)
# texts+=text_splitter3.split_documents(docs3)
texts+=text_splitter4.split_documents(docs4)



# account for deprecation of LLM model
import datetime
# # Get the current date
current_date = datetime.datetime.now().date()
# Define the date after which the 
# model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)
# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
# llm_model = "stabilityai/stablelm-3b-4e1t"  


# conversation_memory = ConversationBufferWindowMemory(buffer_size=10)


# embedding
embeddings = OpenAIEmbeddings()
# storing the texts
vectorstore = FAISS.from_documents(texts, embeddings)

# defining prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. 
If the question is a greeting, then greet the user back.
If the question is a goodbye message, then say "Bye, See you later" to the user back.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Do not repeat the answer. 
Do not mention the chapter number.
Do not mention the S.no number.
Do not mention the document number.
{context}
Question: {question}
Helpful Answer:"""

#prompt
QA_CHAIN_PROMPT = PromptTemplate.from_template(
    template)
#LLM instance
llm = ChatOpenAI(
    temperature = 0, 
    model=llm_model)
#Adding memory
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer')


chat_answer=[]
chat_question=[]

    


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
        input_method = request.form.get('input_method')
        if input_method == 'voice':
            user_question = request.form['transcription']
        else:
            user_question = request.form['user_question']
        formatted_prompt = template.format(context="", question=user_question)
        result = qa({"question": formatted_prompt})
           
        print("Question: ", user_question)
        
        chat_question.append(user_question)
        chat_answer.append(result["answer"])
        print("Answer: ", result['answer'])
        return render_template('index1.html', answer=result['answer'])
    
    return render_template('index1.html')


@app.route('/conversation_history', methods=['GET'])
def chat_history():
    chat_length = len(chat_question)
    return render_template('conversation_history.html', chat_question=chat_question, chat_answer=chat_answer, chat_length=chat_length)

if __name__ == '__main__':
    app.run(debug=True)