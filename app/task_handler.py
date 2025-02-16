import base64
import io
import json
import os
import re
import sqlite3
import subprocess
# import asyncio
from dateutil.parser import parse
from bs4 import BeautifulSoup
import httpx
from PIL import Image
import numpy as np
import pandas as pd
# from torch import cosine_similarity, embedding
import logging

from .llm_client import AI_KEY, AI_URL, embed, get_embeddings, llm_extract_sender, parse_task_with_llm


def process_task(task_description):
    action = parse_task_with_llm(task_description, function_map)
    return action


# :TODO Check this if it works or not.
def format_markdown(file_path, prettier_version):
    # Implementation for formatting markdown
    t = subprocess.run(
        [f"bun x prettier@{prettier_version} --write {file_path}"],
        check=False,
        shell=True,
    )

    print(t.stderr)
    return "Markdown formatted successfully"


def count_weekday(input_file: str, output_file: str, weekday: str = "Wednesday"):
    # Implementation for counting weekdays
    with open(input_file, "r") as file:
        dates = file.readlines()

    wednesdays = sum(
        1
        for date in dates
        if parse(date.strip()).weekday() == 2
    )

    with open(output_file, "w") as file:
        file.write(str(wednesdays))

    return f"Number of Wednesdays: {wednesdays}"


# :TODO Used sh srcipt to do this task. using jq : jq 'sort_by(.last_name, .first_name)' contacts.json > sorted_contacts.json
def sort_contacts(input_file: str, output_file: str):
    # Implementation for sorting contacts
    with open(input_file, "r") as file:
        contacts = json.load(file)

    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_file, "w") as file:
        json.dump(sorted_contacts, file, indent=2)

    return "Contacts sorted successfully"


def recent_log_lines(log_dir: str, output_file: str, count: int):
    # Implementation for getting recent log lines
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    log_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True
    )

    recent_lines = []
    for log_file in log_files[:count]:
        with open(os.path.join(log_dir, log_file), "r") as file:
            recent_lines.append(file.readline().strip())

    with open(output_file, "w") as file:
        file.write("\n".join(recent_lines))

    return "Recent log lines extracted successfully"


def create_markdown_index(docs_dir: str, output_file: str):
    # Implementation for creating markdown index
    index = {}
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    h1_match = re.search(r"^# (.+)$", content, re.MULTILINE)
                    if h1_match:
                        relative_path = os.path.relpath(file_path, docs_dir)
                        index[relative_path] = h1_match.group(1)

    with open(output_file, "w") as file:
        json.dump(index, file, indent=2)

    return "Markdown index created successfully"


def extract_email(input_file: str, output_file: str):
    # Implementation for extracting email
    try:
        with open(input_file, "r") as file:
            email_content = file.read()
    except Exception as e:
        return {"error": "Failed to read the input email file", "details": str(e)}

    try:
        sender_email = llm_extract_sender(email_content)
        if not sender_email:
            return {"error": "LLM returned an empty result for the sender's email"}
    except Exception as e:
        return {"error": "LLM API call failed", "details": str(e)}

    with open(output_file, "w") as file:
        file.write(sender_email)

    return f"Extracted email address: {sender_email}"


def extract_credit_card(image_path: str, output_file: str):
    # Implementation for extracting credit card number
    try: 
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        return "Unable to find image."
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        "Extract the 16 digit credit card number from the following image data. "
        "The data is a base64 encoded representation of an image of a credit card. "
        "Return only the 16 digit credit card number without any spaces or additional text.\n\n"
    )
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    try:
        response = httpx.post(
            AI_URL, 
            headers={"Authorization": f"Bearer {AI_KEY}"},
            json=data, 
            timeout=20
        )
        card_number = response.json()["choices"][0]["message"]["content"].strip().replace(" ", "")
    except Exception as e:
        return  {"error": "LLM API call failed", "details": str(e)}

    # card_number = response.choices[0].text.strip().replace(" ", "")

    with open(output_file, "w") as file:
        file.write(card_number)

    return f"Extracted credit card number: {card_number}"


def find_similar_comments(input_file: str, output_file: str) -> str:
    # Implementation for finding similar comments
    # with open(input_file, "r") as file:
    #     comments = file.readlines()
    
    # embeddings = [embed(comment) for comment in comments]
    # max_similarity = -1
    # most_similar_pair = None

    # for i in range(len(comments)):
    #     for j in range(i + 1, len(comments)):
    #         similarity = cosine_similarity(embeddings[i], embeddings[j])
    #         if similarity > max_similarity:
    #             max_similarity = similarity
    #             most_similar_pair = (comments[i].strip(), comments[j].strip())

    # with open(output_file, "w") as file:
    #     file.write("\n".join(most_similar_pair))

    with open(input_file, "r") as file:
        documents = file.readlines()
    
    documents = [comment.strip() for comment in documents]
    
    try: 
        line_embeddings = get_embeddings(documents)
    except Exception as e: 
        return f"Failed to get embeddings: Cause: {str(e)}"
    similarity_matrix = np.dot(line_embeddings, line_embeddings.T)
    
    np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    
    print(most_similar_indices)
    
    similar_texts = []
    for i in range(len(most_similar_indices)):
        similar_texts.append(documents[most_similar_indices[i]])

    print(similar_texts)
    with open(output_file, "w") as file:
        for text in similar_texts:
            file.write(text + "\n")

    return "Most similar comments written to file"


def calculate_ticket_sales(database_file: str, ticket_type: str = 'Gold', output_file: str = "/data/ticket-sales-gold.txt"):
    # Implementation for calculating ticket sales
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    cursor.execute(f"SELECT SUM(units * price) FROM tickets WHERE type = '{ticket_type}'")
    total_sales = cursor.fetchone()[0]

    conn.close()

    with open(output_file, "w") as file:
        file.write(str(total_sales))

    return f"Total sales for Gold tickets: {total_sales}"


def fetch_api_data():
    # Implementation for fetching API data
    response = httpx.get("https://api.example.com/data")
    data = response.json()

    with open("/data/api_data.json", "w") as file:
        json.dump(data, file, indent=2)

    return "API data fetched and saved successfully"


def git_operations():
    # Implementation for git operations
    repo_url = "https://github.com/example/repo.git"
    repo_path = "/data/git_repo"

    subprocess.run(["git", "clone", repo_url, repo_path])

    with open(f"{repo_path}/new_file.txt", "w") as file:
        file.write("New content")

    subprocess.run(["git", "add", "."], cwd=repo_path)
    subprocess.run(["git", "commit", "-m", "Add new file"], cwd=repo_path)

    return "Git operations completed successfully"


def run_sql_query():
    # Implementation for running SQL queries
    conn = sqlite3.connect("/data/database.db")
    query = "SELECT * FROM users WHERE age > 30"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df.to_csv("/data/query_results.csv", index=False)

    return "SQL query executed and results saved successfully"


def scrape_website():
    # Implementation for scraping website
    url = "https://example.com"
    response = httpx.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string
    paragraphs = [p.text for p in soup.find_all("p")]

    with open("/data/scraped_data.txt", "w") as file:
        file.write(f"Title: {title}\n\n")
        file.write("Paragraphs:\n")
        file.write("\n".join(paragraphs))

    return "Website scraped successfully"


def process_image():
    # Implementation for processing images
    image = Image.open("/data/input_image.jpg")
    resized_image = image.resize((300, 300))
    resized_image.save("/data/resized_image.jpg")

    return "Image processed successfully"


def transcribe_audio():
    # Implementation for transcribing audio
    # Note: This requires additional setup for audio transcription
    audio_file = open("/data/audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    with open("/data/transcript.txt", "w") as file:
        file.write(transcript["text"])

    return "Audio transcribed successfully"


def convert_markdown_to_html():
    # Implementation for converting Markdown to HTML
    with open("/data/input.md", "r") as file:
        markdown_content = file.read()

    html_content = markdown2.markdown(markdown_content)

    with open("/data/output.html", "w") as file:
        file.write(html_content)

    return "Markdown converted to HTML successfully"


def filter_csv():
    # Implementation for filtering CSV
    df = pd.read_csv("/data/input.csv")
    filtered_df = df[df["age"] > 30]

    filtered_df.to_json("/data/filtered_data.json", orient="records")

    return "CSV filtered and converted to JSON successfully"


def run_datagen_script(user_email: str, script_url: str = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"):
    import subprocess
    import sys

    # try:
    #     import uv
    # except ImportError:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
    # Run the datagen.py script with user email as argument
    # subprocess.run(
    #     [
    #         "python3",
    #         "-c",
    #         f'import urllib.request; exec(urllib.request.urlopen("{script_url}").read())',
    #     ],
    #     env={"user_email": user_email},
    # )
    print (user_email, script_url)
    try:
        # t = subprocess.run (f"python3 -f ")
        # subprocess.run(["curl", "-O", script_url])
        t = subprocess.run (f"uv run /app/datagen.py {user_email}", capture_output=True, shell=True, check=False)
        print(t.stdout)
        # t = subprocess.run([f"uv run {script_url} {user_email}"], check=False,shell=True,)
        if t.returncode == 0:
            return "Datagen script executed successfully"
    except Exception as e:
        print (e)
        return "Failed to Gen Data."


# The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt
# A1-10
function_map = {
    "run_datagen_script": run_datagen_script, #failed.
    "format_markdown": format_markdown, # ?
    "count_weekdays": count_weekday, # works
    "sort_contacts": sort_contacts,
    "process_logs": recent_log_lines,
    "create_markdown_index": create_markdown_index,
    "extract_email_address": extract_email,
    "extract_credit_card": extract_credit_card,
    "find_similar_comments": find_similar_comments,
    "calculate_ticket_sales": calculate_ticket_sales,
}
