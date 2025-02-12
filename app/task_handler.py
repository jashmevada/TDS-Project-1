import base64
import datetime
import io
import json
import os
import re
import sqlite3
import subprocess
from bs4 import BeautifulSoup
import httpx
from PIL import Image
import openai
import pandas as pd
from torch import cosine_similarity
from llm_client import parse_task_with_llm


async def process_task(task_description):
    # Use LLM to parse the task and determine the action
    action = parse_task_with_llm(task_description)
    
    # Execute the appropriate action based on the parsed task
    if action == "format_markdown":
        return format_markdown()
    elif action == "count_weekday":
        return count_weekday()
    elif action == "sort_contacts":
        return sort_contacts()
    elif action == "recent_log_lines":
        return recent_log_lines()
    elif action == "create_markdown_index":
        return create_markdown_index()
    elif action == "extract_email":
        return extract_email()
    elif action == "extract_credit_card":
        return extract_credit_card()
    elif action == "find_similar_comments":
        return find_similar_comments()
    elif action == "calculate_ticket_sales":
        return calculate_ticket_sales()
    elif action == "fetch_api_data":
        return fetch_api_data()
    elif action == "git_operations":
        return git_operations()
    elif action == "run_sql_query":
        return run_sql_query()
    elif action == "scrape_website":
        return scrape_website()
    elif action == "process_image":
        return process_image()
    elif action == "transcribe_audio":
        return transcribe_audio()
    elif action == "convert_markdown_to_html":
        return convert_markdown_to_html()
    elif action == "filter_csv":
        return filter_csv()
    else:
        raise ValueError("Unknown task")

# Implement task-specific functions here
def format_markdown():
    # Implementation for formatting markdown
    subprocess.run(["npx", "prettier@3.4.2", "--write", "/data/format.md"])
    return "Markdown formatted successfully"

def count_weekday():
    # Implementation for counting weekdays
    with open("/data/dates.txt", "r") as file:
        dates = file.readlines()
    
    wednesdays = sum(1 for date in dates if datetime.datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)
    
    with open("/data/dates-wednesdays.txt", "w") as file:
        file.write(str(wednesdays))
    
    return f"Number of Wednesdays: {wednesdays}"

def sort_contacts():
    # Implementation for sorting contacts
    with open("/data/contacts.json", "r") as file:
        contacts = json.load(file)
    
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    
    with open("/data/contacts-sorted.json", "w") as file:
        json.dump(sorted_contacts, file, indent=2)
    
    return "Contacts sorted successfully"

def recent_log_lines():
    # Implementation for getting recent log lines
    log_files = [f for f in os.listdir("/data/logs/") if f.endswith(".log")]
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join("/data/logs/", x)), reverse=True)
    
    recent_lines = []
    for log_file in log_files[:10]:
        with open(os.path.join("/data/logs/", log_file), "r") as file:
            recent_lines.append(file.readline().strip())
    
    with open("/data/logs-recent.txt", "w") as file:
        file.write("\n".join(recent_lines))
    
    return "Recent log lines extracted successfully"


def create_markdown_index():
    # Implementation for creating markdown index
    index = {}
    for root, _, files in os.walk("/data/docs/"):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    h1_match = re.search(r'^# (.+)$', content, re.MULTILINE)
                    if h1_match:
                        relative_path = os.path.relpath(file_path, "/data/docs/")
                        index[relative_path] = h1_match.group(1)
    
    with open("/data/docs/index.json", "w") as file:
        json.dump(index, file, indent=2)
    
    return "Markdown index created successfully"


def extract_email():
    # Implementation for extracting email
    with open("/data/email.txt", "r") as file:
        email_content = file.read()
    
    response = openai.create(
        engine="text-davinci-002",
        prompt=f"Extract the sender's email address from the following email:\n\n{email_content}\n\nEmail address:",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    email_address = response.choices[0].text.strip()
    
    with open("/data/email-sender.txt", "w") as file:
        file.write(email_address)
    
    return f"Extracted email address: {email_address}"



def extract_credit_card():
    # Implementation for extracting credit card number
    image = Image.open("/data/credit-card.png")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Extract the credit card number from the following image:\n\n[Image: {image_base64}]\n\nCredit card number:",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    card_number = response.choices[0].text.strip().replace(" ", "")
    
    with open("/data/credit-card.txt", "w") as file:
        file.write(card_number)
    
    return f"Extracted credit card number: {card_number}"

def find_similar_comments():
    # Implementation for finding similar comments
    with open("/data/comments.txt", "r") as file:
        comments = file.readlines()
    
    # Use OpenAI's API to generate embeddings for each comment
    embeddings = [openai.Embedding.create(input=comment, engine="text-embedding-ada-002")["data"][0]["embedding"] for comment in comments]
    
    # Find the most similar pair of comments
    max_similarity = -1
    most_similar_pair = None
    
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (comments[i].strip(), comments[j].strip())
    
    with open("/data/comments-similar.txt", "w") as file:
        file.write("\n".join(most_similar_pair))
    
    return f"Most similar comments written to file"

def calculate_ticket_sales():
    # Implementation for calculating ticket sales
    conn = sqlite3.connect("/data/ticket-sales.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]
    
    conn.close()
    
    with open("/data/ticket-sales-gold.txt", "w") as file:
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

def generate_data():
    ...

# The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt
