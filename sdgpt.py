import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain import OpenAI,ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import json
# 解析base64编码内容
import base64
from PIL import Image
from io import BytesIO

from flask import Flask, request
import requests
import json

app = Flask(__name__)
def stable_diffusion(data):
   # 设置stable diffusion 的端口x
    url = "http://stablediffusion.rejo9.com/sdapi/v1/txt2img"
    response = requests.post(url, json=data)
    print('response',response.text)
    if response.status_code == 200:
        # JSON 格式的字符串
        # 将 JSON 字符串解析为 Python 字典
        data = json.loads(response.text)
        print(data)
        init_images = data.get("images")
        print(init_images)
        return init_images


def chatsd(prompt,art_style=None):
   
    # 翻译员prompt
    translation_prompt = PromptTemplate(
        input_variables=["text"],
        template="I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. My first sentence is {text}",
    )


    # 创建 llm
    llm = OpenAI(temperature=0.9,model_name='text-davinci-003')

    chain = LLMChain(llm=llm,prompt = translation_prompt)
    # 去除分隔符
    # 将输入的prompt继续宁翻译
    trans_result = chain.run(prompt).split('\n')[-1]

    print('翻译内容',trans_result)
    # print(chain.run('一个胸有成竹的女人'))

    # 
    if art_style is None:

        create_prompt = PromptTemplate(
            input_variables=["idea"],
            template="Stable Diffusion is an AI art generation model. \
                Below is a list of prompts that can be used to generate images with Stable Diffusion:\
                    - portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation\
                    - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, \
                        art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon,\
                        steve argyle, winona nelson \
                    - ghost inside a hunted room, loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm,\
                    digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image \
                    - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays,\
                    vivid, beautiful, trending on artstation,huge scene, grass\
                    - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. \
                    rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt\
                    - athena, greek goddess, claudia black, bronze greek armor, owl crown, \
                    d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration \
                    - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant,\
                    highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade,\
                    tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo \
                    - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, \
                    smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha \
                    I want you to write me a list of detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more. IDEA: {idea}",

        )
        chain = LLMChain(llm=llm,prompt = create_prompt)
        prompt_result = chain.run(trans_result)
    else:
        create_prompt = PromptTemplate(
            input_variables=["idea","art_style"],
            template="Stable Diffusion is an AI art generation model. Below is a list of prompts that can be used to generate images with Stable Diffusion:- portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation\
                    - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, \
                        art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, zoltan boros, , howard lyon,steve argyle, winona nelson \
                    - a drawing of a house with a lot of windows, concept art by toriyama sekien, pixiv, maximalism, concept art, official art, maximalist ,\
                    digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image \
                    - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays,\
                    vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski \
                    - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. \
                    rpg portrait\
                    - athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, \
                    d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration \
                    - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant,\
                    highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade,\
                    tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo \
                    - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, \
                    smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha \
                    I want you to write me a detailed prompts exactly about the idea written after IDEA\
                     And I want this prompt's painting type is {art_style} . Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more. IDEA: {idea}")
        # Multiple Inputs
        create_prompt =create_prompt.format(idea=trans_result, art_style=art_style)
        # llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        prompt_result = llm(c)

    
    # chain = LLMChain(llm=llm,prompt = create_prompt)
    # prompt_result = chain.run(trans_result)
    response_list = prompt_result.split('\n')
    # print(response_list)
    response_list_cleaned = [s.strip('- ') for s in response_list if s != '']

    prompt = response_list_cleaned[0]
    print('AI绘画正向prompt',prompt)
    # 设置要发送的数据
    data = {"prompt": prompt,
           "negative_prompt":"lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature,deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy,bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"}
    print(data)
    print(type(data))
    base = stable_diffusion(data=data)
    return base
'''
    # 解码 Base64 图像字符串
    # print(type(init_images[0]))
    decoded_image = base64.b64decode(init_images[0])

    # 打开图像并将其转换为 PNG 格式
    # Open the decoded image using PIL
    image = Image.open(BytesIO(decoded_image))

    # 将 PNG 图像保存到文件中
    image.save("image.png", "PNG")
    # print(response_list_cleaned)
    # # 构建消息队列
    # conversation = ConversationChain(llm=llm, verbose=True)
    # conversation.predict(input="Hi there!")
'''
@app.route("/chatgpt", methods=["POST"])
def main_sdgpt():
    prompt  = request.json.get('prompt')
    # print(request.json)
    # print(type(request.json))
    art_style = request.json.get('art_style')
    base_64 = chatsd(prompt,art_style=art_style)
    return {"image":base_64}
if __name__ == "__main__":
    app.run(debug=True,port=5001)
