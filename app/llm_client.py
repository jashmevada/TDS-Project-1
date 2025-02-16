import json
import httpx
import os
from app.security import is_safe_path
import numpy as np
from app.utils import get_tools


AI_KEY = os.environ.get('AIPROXY_TOKEN')
AI_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def parse_task_with_llm(task_description, function_map):
    
    _mes = [{"role": "user", "content": task_description}]
    json_tools = get_tools()

    response =  httpx.post(
        AI_URL,
        headers={"Authorization": f"Bearer {AI_KEY}"}, json={ "model": "gpt-4o-mini", "messages": _mes, 
                                                             'tools': [{"type": "function", "function": func} for func in json_tools], 'tool_choice': 'required'} )
    
    
    tool_calls = response.json()['choices'][0]['message']['tool_calls']
    
    # print(f"Response: {response.text}")
    
    if not tool_calls:
        return "No valid function found."
    
    results = [] 
    for tool_call in tool_calls:
        res = handle_function_call(tool_call,function_map)
        results.append(res)
    
    return {"results": results}

async def embed(text: str) -> np.ndarray:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {AI_KEY}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        
        embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
        return embeddings
        # return response.json()["data"][0]["embedding"]

def get_embeddings(texts: list[str]):
    try: 
        response =  httpx.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {AI_KEY}"},
            json={"model": "text-embedding-3-small", "input": texts},)
    except Exception as e: 
        print(e)
        raise e
    
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    return embeddings

def handle_function_call(tool_call, function_map):
    print(f"Functions: {tool_call}, {function_map}")
    try:
        function_name = tool_call['function']['name']
        arguments = json.loads(tool_call['function']['arguments'])
             
        if function_name in function_map:
            return function_map[function_name](**arguments)
        else:
            raise ValueError(f"Function {function_name} not found")
            
    except json.JSONDecodeError:
        raise ValueError("Invalid arguments format")
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {str(e)}")

def llm_extract_sender(email_content: str) -> str: 
    prompt = (
        "Extract the sender's email address from the following email content."
        "Return only the email address as plain text.\n\n"
        f"{email_content}"
    )
    
    response = httpx.post(
        AI_URL,
        headers={"Authorization": f"Bearer {AI_KEY}"},
        json= {"model": "gpt-4o-mini", 'messages': [{'role': 'user', 'content': prompt}]}   
    )
    return response.json()["choices"][0]["message"]["content"].strip()