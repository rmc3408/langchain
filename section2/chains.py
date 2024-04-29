from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv
import os
load_dotenv()

llm = OpenAI()

prompt1 = PromptTemplate(
  template="Write a lambda function in {language} that takes {task}",
  input_variables=["language", "task"],
)
prompt2 = PromptTemplate(
  template="Write one test function in {language} using code:\n{code}",
  input_variables=["language", "task"],
)

chain1 = LLMChain(
  llm=llm,
  prompt=prompt1,
  output_key="code",
)
chain2 = LLMChain(
  llm=llm,
  prompt=prompt2,
)

combined_chain = SequentialChain(
  chains=[chain1, chain2],
  input_variables=["language", "task"],
  output_variables=["code", "text"]
)

res = combined_chain.invoke({ 
    "language": "javascript",
    "task": "a list of 3 integers and returns the sum of all the integers"
})

print(res)