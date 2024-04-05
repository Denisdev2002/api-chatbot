from fastapi import FastAPI
from pydantic import BaseModel
from rag import chain

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




QUESTIONS = []


class Question(BaseModel):
    """Classe de question"""
    question: str
    
async def generate_text(question: str) -> str:
    try:
        result = chain.invoke({'question': question})
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=len(str(e)))

@app.get("/questions", tags=["questions"])
def get_questions() -> list:
    """Listar questões"""
    return QUESTIONS
@app.get("/questions/{question_id}", tags=["questions"])
def get_question(question_id: int) -> dict:
    """Pegar questão"""
    for question in QUESTIONS:
        if question["id"] == question_id:
            return question
    return {}

@app.post("/questions", tags=["questions"])
async def set_question(question: Question) -> dict:
    """Criar questão"""
    question_dict = question.dict()
    question_dict["id"] = len(QUESTIONS) + 1
    generated_text = await generate_text(question.question)
    question_dict["generated_text"] = generated_text
    QUESTIONS.append(question_dict)
    return question_dict

@app.put("/questions/{question_id}", tags=["questions"])
def put_question(question_id: int, question: Question) -> dict:
    """Atualizar question"""
    for index, quest in enumerate(QUESTIONS):
        if quest["id"] == question_id:
            QUESTIONS[index] = question
            return question
    return {}

@app.delete("/questions/{question_id}", tags=["questions"])
def delete_question(question_id: int) -> dict:
    """Remover questão"""
    for index, quest in enumerate(QUESTIONS):
      if quest["id"] == question_id:
            QUESTIONS.pop(index)
            return {"message": "Questão removida com sucesso "}
    return {}