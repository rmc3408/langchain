from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from models.doc import Doc, DocResponse
from typing import Optional

from prompt import qa_template, generate_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI()


app = FastAPI()


docs: [Doc] = []


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def logging(request, call_next):
    response = await call_next(request)
    return response


# ROUTES
@app.post("/doc/conversation")
async def read(query: str):
    # User collects all information from Database
    text_ocr = 'This is a user has 3 years of intense fitness, diet of 3000 calories per day and his goal is bulking'

    prompt = ChatPromptTemplate.from_template(qa_template)

    chain = prompt | chat_model
    result = chain.invoke({
        "context": generate_context(text_ocr),
        "question": query
    })
    return result


@app.get("/doc")
async def all_docs(completed: Optional[bool] = False):
    if completed is None:
        return docs
    else:
        return [doc for doc in docs if doc.isCompleted is True]


@app.get("/doc/{item}")
async def one_doc(item):
    for doc in docs:
        if doc.id == item:
            return doc

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Cannot FIND the scanned doc item"})


async def email(doc: Doc):
    print(f"Email notify {doc.id} and sent")


@app.post("/doc", response_model=DocResponse)
async def create_doc(doc: Doc, back_doc: BackgroundTasks) -> Doc:
    doc.id = len(docs) + 1
    back_doc.add_task(email, doc)
    docs.append(doc)
    return doc


@app.put("/doc/{id}")
async def update_doc(id: int, new_doc: Doc):
    for index, doc in enumerate(docs):
        if doc.id == id:
            docs[index] = new_doc
            docs[index].id = id
            return JSONResponse({"message": "content updated"}, status_code=status.HTTP_202_ACCEPTED)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Cannot UPDATE Scanned doc item"})


@app.delete("/doc/{id}")
async def delete_doc(id: int):
    for index, doc in enumerate(docs):
        if doc.id == id:
            del docs[index]
            return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Cannot DELETE scanned doc item"})



