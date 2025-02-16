# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

os.environ['TRANSFORMERS_CACHE'] = r'J:\Python\IITM_python\TDS\Project-1\models\cache'

tokenizer = AutoTokenizer.from_pretrained("../models/TAID-LLM-1.5B")
model = AutoModelForCausalLM.from_pretrained("../models/TAID-LLM-1.5B")

messages = [
    {"role": "user", "content": "Who are you?"},
]

email_content = ''
with open("../data/email.txt") as f:
    email_content = f.read()
    
inputs = tokenizer(f"Extract the only sender's email address from the following email:\n\n{email_content}\n\nEmail address:", return_tensors="pt", return_token_type_ids=False)
# output = model(**inputs)
response = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])