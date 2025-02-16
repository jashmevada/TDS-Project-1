from fastapi import FastAPI, HTTPException, Query
from app.task_handler import process_task
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    result = process_task(task)
    return result

@app.get("/read")
async def read_file(path: str):
    from app.security import is_safe_path
    if is_safe_path(path):
        try:
            with open(path, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            return HTTPException(status_code=5000, detail="Not able to open File.")
    return HTTPException(status_code=5000, detail="Error executing function.")
