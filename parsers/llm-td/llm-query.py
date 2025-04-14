import sys, time, os

from together import Together
from openai import OpenAI

def get_client(model):
    if "gpt" in model:
        return OpenAI()
    elif "deepseek-ai/DeepSeek-R1" in model:
        return Together(api_key=os.environ["TOGETHER_API_KEY"])
    elif "deepseek-reasoner" in model:
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
model=sys.argv[1]
client = get_client(model)

prompt = ""

for line in sys.stdin:
  prompt += line

invoc_start = time.time()
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=4096,
    temperature=0.0
).choices[0].message.content
invoc_end = time.time()

with open("invoc_time.txt", "a") as f:
  f.write(f"{str(invoc_end - invoc_start)}\n")
  f.close()

print(response)
