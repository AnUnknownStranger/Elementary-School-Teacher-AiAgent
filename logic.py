import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import glob

#Initialize the Model with api_key
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.4, 
    api_key="sk-b03ebd023fa744daa62e8d2c0d899111" 
)
# Initialize an empty conversation history.
chat_history = []

#Execute python scripts/preprocess.py and python scripts/index.py in terminal before executing this section
def get_retriever():
    #Perform embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector", embeddings, allow_dangerous_deserialization=True)  
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = get_retriever()

#Setting up a different prompt for math to guide students step by step instead of directly providing answers
class MathAgent:
    def __init__(self,llm):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Grade {grade} Math Socratic Tutor.\n"
                "### CRITICAL RULE: DO NOT PROVIDE DIRECT ANSWERS OR FINAL NUMERICAL RESULTS. ###\n"
                "If you state the answer directly, you have failed task.\n\n"
                "YOUR GOAL: Act as a coach. Guide the student through ONE small step at a time.\n"
                "1. Identify where the student currently is in the process.\n"
                "2. Explain how they should approach the NEXT step (method, not result).\n"
                "3. Do NOT perform calculations for them.\n"
                "### FORMATTING RULES ###\n"
                "• Use standard LaTeX formatting wrapped in single dollar signs ($...$).\n"
                "• Example: $\\frac{{1}}{{2}} \\times \\frac{{3}}{{4}}$\n\n"
                "5. Only confirm the final result AFTER the student provides it.\n\n"
                "TEXTBOOK CONTEXT:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()
    def handle(self, question,context,grade, history):
        return self.chain.invoke({"question": question, "context": context,"grade": grade, "history": history})
#Explain the english concepts in a simple way while defining the difficult vocabs
class EnglishAgent:
    def __init__(self,llm):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Grade {grade} English & Literacy Coach.\n"
                "INSTRUCTIONS:\n"
                "• Explain literary elements (theme, character, setting, plot) in simple language.\n"
                "• Define difficult vocabulary words and use them in new example sentences.\n"
                "• Provide a short example (e.g., a sample sentence or paragraph).\n\n"
                "TEXTBOOK CONTEXT:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()
    def handle(self, question,context,grade, history):
        return self.chain.invoke({"question": question, "context": context,"grade": grade, "history": history})
#Explain the science concepts with a simple everyday example and explain the complex vocabs
class ScienceAgent:
    def __init__(self,llm):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a fun Grade {grade} Science Teacher. Focus on deep conceptual understanding.\n"
                "INSTRUCTIONS:\n"
                "• Explain the concept using a relatable real-world story or everyday example.\n"
                "• Break down cause-and-effect relationships clearly.\n"
                "• Define scientific vocabulary in simple terms.\n"
                "• Describe a simple experiment the student could imagine or try safely.\n\n"
                "TEXTBOOK CONTEXT:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()
    def handle(self, question,context,grade, history):
        return self.chain.invoke({"question": question, "context": context,"grade": grade, "history": history})
#Explain the historical events using stories and complex vocabs in simple terms
class SocialStudiesAgent:
    def __init__(self,llm):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Grade {grade} Social Studies Historian.\n"
                "INSTRUCTIONS:\n"
                "• Explain historical events or geography using a story-based approach.\n"
                "• Clarify timelines and why the topic matters today.\n"
                "• Define terms like democracy, economy, or culture in simple terms.\n\n"
                "TEXTBOOK CONTEXT:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()
    def handle(self, question,context,grade, history):
        return self.chain.invoke({"question": question, "context": context,"grade": grade, "history": history})

#This is a general agent that answers question in simple ways
class GeneralAgent:
    def __init__(self,llm):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful Grade {grade} School Assistant.\n"
                "1. Define any tricky vocabulary words.\n"
                "2. Give a clear final summary in 2-3 sentences.\n"
                "3. Suggest 3 questions for the students to further consider.\n\n"
                "TEXTBOOK CONTEXT:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()
    def handle(self, question,context,grade, history):
        return self.chain.invoke({"question": question, "context": context,"grade": grade, "history": history})

classifier_prompt = ChatPromptTemplate.from_template(
    "Based on the student question. Detect the subject (Math, Science, Social Studies or English) or is it a general question"
    "and the grade level (1, 2, 3, 4, or 5).\n"
    "Remeber that the location is USA based!!!\n"
    "Return JSON only: {{\"subject\": \"...\", \"grade\": ...}}\n"
    "Question: {question}"
)
def Route_Question(student_question, history_list):
    #Have Agent classify the grade level and the subject of the question
    classifier_chain = classifier_prompt | llm | JsonOutputParser()
    student_info = classifier_chain.invoke({'question':student_question})
    subject = student_info.get('subject','general')
    grade = student_info.get('grade','Elementary')
    #Fetch the relevant context based on teh grade level, subject, and student question
    RAGdata = retriever.invoke(f"Grade {grade} {subject}: {student_question}")
    referenced_sources = list(set([d.metadata['source'] for d in RAGdata if 'source' in d.metadata]))
    context = "\n\n".join([f"Source [{d.metadata['source']}]: {d.page_content}" for d in RAGdata])
    #Select Agent based on the subjects
    agents = {'math':MathAgent(llm),
               'english':EnglishAgent(llm),
               'science':ScienceAgent(llm),
               'social studies':SocialStudiesAgent(llm),
               'socialstudies':SocialStudiesAgent(llm)}
    response = agents.get(subject.lower(),GeneralAgent(llm)).handle(student_question,context,grade,history_list)
    #Store the results in chat history
    history_list.append(HumanMessage(content=student_question))
    history_list.append(AIMessage(content=response))
    
    return {
        "detected_subject": subject,
        "detected_grade": grade,
        "response": response,
        "sources": referenced_sources
    }



