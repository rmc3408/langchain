from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages")
])

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=model,
)

chain = prompt | model

while True:
    user_input = input(">> ")
    actual = HumanMessage(content=f"{user_input}")
    memory.chat_memory.add_user_message(actual)
    result = chain.invoke({
        "messages": memory.chat_memory.messages,
    })
    memory.chat_memory.add_ai_message(result)
    print(result.content)
