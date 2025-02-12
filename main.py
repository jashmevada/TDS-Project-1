from fastapi import FastAPI, Query
from app.task_handler import process_task

app = FastAPI()

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    result = await process_task(task)
    return result

@app.get("/read")
async def read_file(path: str):
    from app.utils import read_file_content
    return read_file_content(path)
