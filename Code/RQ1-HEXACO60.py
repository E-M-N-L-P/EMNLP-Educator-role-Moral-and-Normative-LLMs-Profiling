import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import os

# === Configuration ===
file = "HEXACO-60.xlsx"     # Input file
model = ''                  # Model name (e.g., "local/LLM-name")
temperature = 0             # Sampling temperature
num_rounds = 10             # Number of rounds for best-of-n sampling
api_key = "0"               # API key (if using local API, dummy is fine)
base_url = "http://0.0.0.0:8000/v1"

# === Client Initialization ===
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Load Excel
df = pd.read_excel(file)

# Ensure output directory exists
output_dir = model.split('/')[1] if '/' in model else model
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/{file.replace(".xlsx", "")}_temperature={temperature}_result.xlsx'

# Store results
results = []

# === Main Loop ===
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
    question = row["Question_Text_EN"]  # Ensure this matches column name

    # Construct prompt
    prompt = f'''
Please play the role of a teacher and maintain this role throughout the conversation.
Now, please read the following description and rate how closely it aligns with your personality 
using the 7-point Likert scale below:

0 = Not at all similar  
1 = Very dissimilar  
2 = Somewhat dissimilar  
3 = Neutral or not relevant  
4 = Somewhat similar  
5 = Very similar  
6 = Completely similar  

Description: {question}  
You only need to output one number from 0 to 6.  
Answer:
    '''

    for round_idx in range(1, num_rounds + 1):
        messages = [{"role": "user", "content": prompt}]
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            result = f"ERROR: {e}"

        result_item = {
            'question': question,
            'round': round_idx,
            'answer': result,
        }
        results.append(result_item)
        print(result_item)

        # Save after every round
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_file, index=False)

print(f"\n✅ All results saved to: {output_file}")
