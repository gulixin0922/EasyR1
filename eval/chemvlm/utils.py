
import base64
import time
import os
from openai import OpenAI, DefaultHttpxClient

if os.getenv('OPENAI_BASE_URL',None):
    Client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL') # 使用自定义的HTTP客户端
        )
else:
    proxies = {
        "http://": os.getenv('OPENAI_PROXY_URL'),
        "https://": os.getenv('OPENAI_PROXY_URL')
    }

    http_client = DefaultHttpxClient(proxies=proxies)

    print(f"OPENAI_API_KEY:{os.getenv('OPENAI_PROXY_URL')}")
    print(f"OPENAI_API_KEY:{os.getenv('OPENAI_API_KEY')}")

    Client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        http_client=http_client # 使用自定义的HTTP客户端
        )

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def openai_chat(client,
                model_name,
                question,
                generate_config={},
                image=None,
                sys_prompt=None):
    
    messages = [{
                    'role':'user',
                    'content': [{
                        'type': 'text',
                        'text': question,
                    },],
                }]
    if sys_prompt:
        messages = [{
                    "role":"system",
                    "content":sys_prompt 
        }] + messages
    # start_time = time.time()
    if image:
        base64_images  = [encode_image(image=img) for img in image]
        for base64_image in base64_images:
            messages[-1]["content"].append({
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{base64_image}",
                        "max_dynamic_patch": 12
                    }
                })
    count = 0
    while count < 6:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generate_config,
                timeout=180
            )
            return response.choices[0].message.content
        
        except Exception as e:
            count += 1
            print(e)
            time.sleep(1)
