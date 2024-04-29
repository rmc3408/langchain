from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from operator import itemgetter

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--language", default="typescript")
parser.add_argument("--numbers", default="5-10")
parser.add_argument("--operation", default="sum of all the numbers")
args = parser.parse_args()

from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("OPENAI_API_KEY"))

output_parser = StrOutputParser()
model = ChatOpenAI()

prompt1 = ChatPromptTemplate.from_messages([
  ("system", "Write a simple {language} function that a list of number from {numbers} and return {operation}]"),
])
prompt2 = ChatPromptTemplate.from_template("Add {number} to the result of {function} and give total of the result.")


chain1 = prompt1 | model #| output_parser
chain2 = { "function": chain1, "number": itemgetter("number") } | prompt2 | model #| output_parser



chain2_result = chain2.invoke({
  "language": args.language,
  "numbers": args.numbers,
  "operation": args.operation,
  "number": "5",
})

print(chain2_result.content)
