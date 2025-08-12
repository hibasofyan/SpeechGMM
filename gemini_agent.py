import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain

class GeminiAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("La clé API Google n'est pas définie dans le fichier .env")
        try:
            with open('agent_identity.txt', 'r', encoding='utf-8') as f:
                self.agent_identity = f.read()
        except Exception as e:
            print(f"Erreur lors du chargement de l'identité de l'agent : {str(e)}")
            self.agent_identity = "Je suis AryadAI, un assistant IA conversationnel."
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            convert_system_message_to_human=True
        )
        
        self.normal_system_message = f"""{self.agent_identity}

Tu dois :
1. Répondre de manière naturelle, courte et engageante, sans trop détailler.
2. Maintenir le contexte de la conversation.
3. Fournir des réponses précises et utiles.
4. Adapter ton ton au contexte de la conversation.
5. Toujours te présenter comme AryadAI quand on te demande ton nom ou ton identité.

Exemple de conversation :

Utilisateur : Bonjour  
AryadAI : Bonjour, comment tu vas ?

Utilisateur : Bien, et toi ?  
AryadAI : Moi aussi, merci. Tu as prévu quoi aujourd'hui ?

Utilisateur : Jouer au foot.  
AryadAI : Super, profite bien !
"""
        self.interpreter_system_message = """Tu es un interprète professionnel. Tu dois :
        1. Détecter automatiquement la langue source
        2. Traduire le message dans la langue cible ({target_language}) de manière littérale
        3. Ne rien ajouter ni retirer du message original
        4. Ne pas fournir d'explications supplémentaires
        5. Ne pas modifier le ton ou le style du message
        6. Retourner UNIQUEMENT la traduction, sans aucun autre texte ni formatage

        IMPORTANT : Ne pas ajouter de texte comme "traduccion_literal" ou autre. Retourner uniquement la traduction."""
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.normal_system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
        self.interpreter_chain = None
        
    def get_response(self, message):
        try:
            if self.interpreter_chain:
                response = self.interpreter_chain.predict(input=message)
            else:
                response = self.chain.predict(input=message)
            return response
        except Exception as e:
            return f"Erreur: {str(e)}"
    
    def update_prompt_for_interpreter(self, target_language):
        interpreter_prompt = ChatPromptTemplate.from_messages([
            ("system", self.interpreter_system_message.format(target_language=target_language)),
            ("human", "{input}")
        ])
        self.interpreter_chain = LLMChain(
            llm=self.llm,
            prompt=interpreter_prompt,
            verbose=True
        )
    
    def restore_normal_prompt(self):
        self.interpreter_chain = None
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.normal_system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def reset_memory(self):
        self.memory.clear()

    def reset_chat(self):
        self.memory.clear() 