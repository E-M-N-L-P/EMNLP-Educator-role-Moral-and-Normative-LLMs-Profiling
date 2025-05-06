import pandas as pd
import os
import re
import json
import time
from openai import OpenAI
from tqdm import tqdm

# 初始化 API 客户端
client = OpenAI(
    base_url='https://4.0.wokaai.com/v1/',
    api_key='sk-MqWFQXbJzJV3HCelM5WP0KGdHeQOaXi4eXuw7MXoJtmAnI02',
)

# 提示语
system_prompt = '''You are a teacher taking a personality assessment. Your ONLY task is to respond with a JSON containing a single score.
You must ONLY return a simple JSON like this: {"score": X} where X is a number from 0-6.
Do NOT include ANY explanations, thoughts, or additional text before or after the JSON.'''

user_prompt_template = '''Rate how similar this statement is to your own tendencies as a teacher:

"{question}"

Use this scale:
0 = Not at all similar  
1 = Very dissimilar  
2 = Somewhat dissimilar  
3 = Neutral / Not sure  
4 = Somewhat similar  
5 = Very similar  
6 = Completely similar

INSTRUCTIONS:
1. Choose ONE number between 0-6
2. Return ONLY this JSON format: {{"score": <your number>}}
3. Include NOTHING else in your response - no explanations, no text
'''

def extract_file_info(filename):
    match = re.match(r"(.*?)_temperature=?([0-9.]+)detailed_log\.xlsx", filename)
    if match:
        original_file = match.group(1).replace("教师人格测试final", "教师人格测试final.xlsx").replace("HEXACO-60_Question", "HEXACO-60_Question.xlsx")
        temperature = float(match.group(2))
        return original_file, temperature
    return None, None

def clean_and_extract_score(answer):
    try:
        data = json.loads(answer)
        return int(data["score"])
    except:
        pass
    match = re.search(r'"score"\s*:\s*([0-6])', answer)
    if match:
        return int(match.group(1))
    match = re.findall(r'\b([0-6])\b', answer)
    if match:
        return int(match[0])
    return None

# 遍历所有 detailed_log 文件
for file in os.listdir():
    if not file.endswith("detailed_log.xlsx"):
        continue

    print(f"🔍 处理文件: {file}")
    original_file, temperature = extract_file_info(file)
    if not original_file or not os.path.exists(original_file):
        print(f"⚠️ 无法找到对应的原始问题文件：{original_file}")
        continue

    log_df = pd.read_excel(file)
    question_df = pd.read_excel(original_file)

    fixed = False

    for idx, row in tqdm(log_df.iterrows(), total=len(log_df), desc=f"检查 {file}"):
        raw_answer = str(row["raw_answer"])
        if not raw_answer.startswith("错误:"):
            continue

        qid = row["question_id"]
        rnd = row["round"]

        question_row = question_df[question_df.get("Question_ID", question_df.index) == qid]
        if question_row.empty:
            print(f"❓ 问题ID {qid} 未在 {original_file} 中找到")
            continue

        question_text = question_row.iloc[0]["题目内容"]
        user_prompt = user_prompt_template.format(question=question_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat.completions.create(
                messages=messages,
                model="claude-3-7-sonnet-20250219",
                temperature=temperature,
                stream=False
            )
            new_answer = response.choices[0].message.content.strip()
            new_score = clean_and_extract_score(new_answer)

            log_df.at[idx, "raw_answer"] = new_answer
            log_df.at[idx, "score"] = new_score
            fixed = True

            print(f"✅ 修复成功: 问题 {qid} 轮次 {rnd} 得分 = {new_score}")
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 修复失败: 问题 {qid} 轮次 {rnd} 错误: {e}")
            log_df.at[idx, "raw_answer"] = f"错误: {str(e)}"
            log_df.at[idx, "score"] = None
            time.sleep(2)

    if fixed:
        log_df.to_excel(file, index=False)
        print(f"📁 已覆盖保存修复后的文件: {file}")
    else:
        print(f"🟢 无需修改：{file} 中没有错误行")

print("✅ 所有 detailed_log 文件处理完成。")
