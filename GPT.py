# 1. Try generating with command K. Ask for a pytorch script of a feedforward neural network
# 2. Then, select the outputted code and hit chat. Ask if there's a bug. Ask how to improve.
# 3. Try selecting some code and hitting edit. Ask the bot to add residual layers.from transformers import AutoTokenizer, AutoModelForCausalLM
#!/usr/bin/python
# -*- coding: utf-8 -*-



from flask import Flask, request
import requests
import json
app = Flask(__name__)


API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "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"
proxy = { "http": "http://127.0.0.1:10809", "https":"https://127.0.0.1:10809"}
#请帮我分析出以下prompt中以逗号分隔的每一个短句分别是属于什么描写类型的，以[{'短句'： ，'类型'：'**类型' }]]的格式输出。"
#请帮我分析出以下词汇是属于什么描述类型的，以[{'短句'： ，'类型'：'**类型' }]]的格式输出。
#我想让你充当数据标注员，我会给你一段结构化格式的客服销售记录，请帮我标注出该对话记录的寒暄阶段，商品介绍阶段等各种阶段分别是哪几句，最后把标注结果以 [{'text': , 'id': , 'stage':'**阶段' }，{最终结果：''}] 的格式用中文返回给我。

# 请帮我翻译一下这段话

cache = {}

def chatgpt(prompt):
    if prompt in cache:
        return cache[prompt]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "我想让你充当中译英的翻译员，我会给你一段话，请我，最后把标注结果以 [{'origin': , 'result':'**阶段' }] 的格式用中文返回给我。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response_text = response.json().get("choices")[0]["message"]["content"]
    
    cache[prompt] = response_text
    
    return response_text

@app.route("/translate", methods=["POST"])
def chatgpt_endpoint():
    prompt = request.json.get("prompt")
    response_text = chatgpt(prompt)
    return {"response_text": response_text}

if __name__ == "__main__":
    app.run(debug=True,port=5001)