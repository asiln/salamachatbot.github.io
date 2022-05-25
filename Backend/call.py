from talk import chat
from fastapi import FastAPI
import uvicorn
app = FastAPI(debug=True)

@app.get("/")
def chatbot(You: str):
    return {"Response":chat(You)}

if __name__ == "__main__":
    uvicorn.run(app)
