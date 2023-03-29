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
            {"role": "system", "content": "Stable Diffusion is an AI art generation model. Below is a list of prompts that can be used to generate images with Stable Diffusion: - portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson - ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt - athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha I want you to write me a list of detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more. IDEA: "},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response_text = response.json().get("choices")[0]["message"]["content"]
    
    cache[prompt] = response_text
    
    return response_text

@app.route("/genprompt", methods=["POST"])
def chatgpt_endpoint():
    prompt = request.json.get("prompt")
    response_text = chatgpt(prompt)
    return {"response_text": response_text}

if __name__ == "__main__":
    app.run(debug=True,port=5002)