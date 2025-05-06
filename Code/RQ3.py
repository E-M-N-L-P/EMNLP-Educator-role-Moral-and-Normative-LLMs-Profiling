import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os

# Configuration
file = "RQ3_prompts.xlsx"  # The moral dilemma Excel file for RQ3
model_id = ''
output_dir = model_id.split('/')[-1] if '/' in model_id else model_id
os.makedirs(output_dir, exist_ok=True)

api_keys = [
    '0',
]

# Load the moral dilemmas from RQ3
df = pd.read_excel(file)
results = []  # Store the model responses

# Define temperature values
temperatures = [0.25, 0.5, 0.75, 1.0]  # Selected temperature settings

# Outer loop: iterate through temperatures
for temperature in temperatures:
    print(f"\n🌀 Processing temperature {temperature}...\n")

    # Iterate through each moral dilemma
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing dilemmas (temp={temperature})"):
        prompt = row["Prompt"]  # Soft prompt
        student_statement = row["Student Statement"]  # Student's input
        dimension = row["Dimension"]  # Moral dilemma dimension

        # Construct full prompt
        full_prompt = f"""
        Role: {prompt}
        Please play the role of a teacher described above, and maintain this role throughout the conversation. 
        Now, read the following statement from the student and respond accordingly. Provide your full response to the student.
        Student's Statement: {student_statement}
        Answer:
        """

        for api_key in api_keys:
            try:
                print(f"Trying API with key: {api_key}")

                # Connect to the local ModelScope-compatible API
                client = OpenAI(
                    base_url='http://0.0.0.0:8000/v1',
                    api_key=api_key,
                )

                # Prepare model input
                messages = [{"role": "user", "content": full_prompt}]
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    stream=True  # Streaming enabled
                )

                done_reasoning = False
                reasoning, answer = '', ''
                for chunk in response:
                    # Check if chunk contains valid content
                    if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta'):
                        # Extract reasoning content if present
                        reasoning_content = getattr(chunk.choices[0].delta, 'reasoning_content', None)
                        if reasoning_content:
                            reasoning += reasoning_content

                        # Extract answer content if present
                        content = getattr(chunk.choices[0].delta, 'content', None)
                        if content:
                            answer += content
                            if not done_reasoning:
                                done_reasoning = True

                # Store result
                result_item = {
                    'Index': index + 1,
                    'Dimension': dimension,
                    'Reasoning': reasoning if reasoning else "NULL",
                    'Answer (English)': answer.strip(),
                }
                results.append(result_item)

                # Print output
                print(result_item)
                print(f"[Raw Output] {answer}\n")

                # Save results in real-time
                results_df = pd.DataFrame(results)
                output_file = f'{output_dir}/{file.replace(".xlsx", "")}_temperature={temperature}_result.xlsx'
                results_df.to_excel(output_file, index=False)

                break

            except Exception as e:
                continue

    print(f"\n✅ Results for temperature {temperature} saved to: {output_file}")

print(f"\n✅ All results have been saved to the final output: {output_file}")
