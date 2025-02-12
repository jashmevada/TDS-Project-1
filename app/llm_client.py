import os
import httpx
from sentence_transformers import SentenceTransformer

AI_KEY = os.environ.get("AI_KEY")


list_of_tasks = [
    "format_markdown", "count_weekday", "sort_contacts", "recent_log_lines", "create_markdown_index",
    "extract_email", "extract_credit_card", "find_similar_comments", "calculate_ticket_sales", "fetch_api_data", 
    "git_operations", "run_sql_query", "scrape_website", "process_image", "transcribe_audio", "convert_markdown_to_html",
    "filter_csv", "generate_data"
]


def parse_task_with_llm(task_description):
    messages=[
        {"role": "assistant", "content": f"You are a helpful assistant. Please Return Action from given list when asked.\nList of Action:{list_of_tasks}"},
        {
            "role": "user",
            "content": f"Parse the following task and return the corresponding action:\n\n{task_description}\n\nAction:"
        }
        ]

    response =  httpx.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {AI_KEY}"}, json={ "model": "gpt-4o-mini", "messages": messages } )
    

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error"
    
async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {AI_KEY}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        return response.json()["data"][0]["embedding"]

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('../models/models--sentence-transformers--all-MiniLM-L6-v2') #all-mpnet-base-v2
# model.save("../models/llm_model")
embeddings = model.encode(sentences).tolist()
print(embeddings)