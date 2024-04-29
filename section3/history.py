from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Calculate the operations"),
    MessagesPlaceholder(variable_name="messages")
])

history = ChatMessageHistory()  # Temporary memory history
history.add_ai_message("Starting with resultcd section3 is 0")

chain = prompt | model

while True:
    user_input = input(">> ")
    actual = HumanMessage(content=f"{user_input}")
    history.add_user_message(actual)
    result = chain.invoke({
        "messages": history.messages,
    })
    history.add_ai_message(result)
    print(result.content)
