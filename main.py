import requests

url = 'http://127.0.0.1:5001/translate'
# 传递一个中文的名词
key = input('请输入描绘的主题')
data = {'prompt': key}

response = requests.post(url, json=data)

trans_result = response.json()['response_text']

# 将翻译结果传递给 prompt 生成器
trans_data = eval(trans_result)[0]['result']
prompt_data = {'prompt':trans_data}
url2 = 'http://localhost:5002/genprompt'

# response = requests.post(url, json=prompt_data)

response_prompt = requests.post(url2, json=prompt_data)

trans_result = response_prompt.json()

# json_data = json.loads(json_str)
response_text = trans_result['response_text']
response_list = response_text.split('\n')
response_list_cleaned = [s.strip('- ') for s in response_list]

print(response_list_cleaned)
