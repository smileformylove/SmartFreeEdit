import base64
import re
import torch

from PIL import Image
from io import BytesIO
import numpy as np
import gradio as gr

from openai import OpenAI

from SmartFreeEdit.gpt4_o.instructions import (
    create_editing_category_messages_gpt4o, 
    create_ori_object_messages_gpt4o, 
    create_add_object_messages_gpt4o,
    create_editing_remove,
    create_editing_replace,
    create_editing_add,
    create_apply_editing_messages_gpt4o,
    detect_main_object_messages_gpt4o)

from SmartFreeEdit.utils.utils_lisa import run_grounded_sam

import requests
import json

# Azure OpenAI API details
# API_KEY = '04992b6701a34d9297fe2f94611a2e22'
# API_VERSION = "2024-08-01-preview"
# END_POINT = 'https://06.openai.azure.com/'
# ENGINE = "4o"

# Construct the full URL for Azure OpenAI
#url = f"{END_POINT}/openai/deployments/{ENGINE}/chat/completions?api-version={API_VERSION}"

def encode_image(img):
    img = Image.fromarray(img.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def run_azure_openai_inference(url, gpt_key, messages):
    payload = json.dumps({
        "messages": messages
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': gpt_key,
    }
    print('Waiting for GPT-4 response ...')
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=20)
        response.raise_for_status()  # 检查请求是否成功
        response_str = response.json()['choices'][0]['message']['content']
        return response_str
    except requests.exceptions.Timeout:
        raise gr.Error("Request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"An error occurred: {e}")

# def run_gpt4o_vl_inference(vlm_model, 
#                            messages):
#     response = vlm_model.chat.completions.create(
#         model="gpt-4o-2024-08-06",
#         messages=messages
#     )
#     response_str = response.choices[0].message.content
#     return response_str

### response editing type
def vlm_response_editing_type(url, 
                              gpt_key, 
                              image, 
                              editing_prompt,
                              device):

    
    messages = create_editing_category_messages_gpt4o(editing_prompt)
    response_str = run_azure_openai_inference(url, gpt_key, messages)
    
    try:
        for category_name in ["Addition","Remove","Local","Global","Background", "Resize"]:
            if category_name.lower() in response_str.lower():
                return category_name
    except Exception as e:
        raise gr.Error("Please input OpenAI API Key. Or please input correct commands, including add, delete, and modify commands. If it still does not work, please switch to a more powerful VLM.")



### response object to be edited        
def vlm_response_object_wait_for_edit(url, 
                                      gpt_key, 
                                      image, 
                                      category, 
                                      editing_prompt,
                                      device):
    if category in ["Global", "Addition"]:
        edit_object = "nan"
        return edit_object
    elif category=="Background":
        edit_object = "background"
        return edit_object

    messages = create_ori_object_messages_gpt4o(editing_prompt)
    response_str = run_azure_openai_inference(url, gpt_key, messages)
    return response_str


### response mask
def vlm_response_mask(url,
                      gpt_key,
                      category, 
                      image, 
                      editing_prompt, 
                      object_wait_for_edit, 
                      lisa_model,
                      tokenizer,
                      device="cuda",
                      boxs=None,
                      ):
    mask = None
    if editing_prompt is None or len(editing_prompt)==0:
        raise gr.Error("Please input the editing instruction!")
    height, width = image.shape[:2]
    if category=="Addition":
        try:
            base64_image = encode_image(image)
            messages = create_add_object_messages_gpt4o(boxs, editing_prompt, base64_image, height=height, width=width)
            response_str = run_azure_openai_inference(url, gpt_key, messages)
            pattern = r'\[\d{1,3}(?:,\s*\d{1,3}){3}\]'
            box = re.findall(pattern, response_str)
            box = box[0][1:-1].split(",")
            for i in range(len(box)):
                box[i] = int(box[i])
            cus_mask = np.zeros((height, width))
            cus_mask[box[1]: box[1]+box[3], box[0]: box[0]+box[2]]=255
            mask = cus_mask
        except:
            raise gr.Error("Please set the mask manually, currently the VLM cannot output the mask!")

    elif category=="Background":
        labels = "background"
    elif category=="Global":
        mask = 255 * np.ones((height, width))
    else:
        labels = object_wait_for_edit
    
    if mask is None:
        try:
            reasoning_prompt = f"Please segment the object: {labels}"
            mask = run_grounded_sam(
                input_image={"image":image, "mask":None}, 
                text_prompt=reasoning_prompt, 
                task_type="segmentation", 
                model=lisa_model,
                tokenizer=tokenizer,
                device=device,
            )
            print(f"Detections: {mask}")
            print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
        except:
            raise gr.Error("Please set the mask manually, currently the VLM cannot output the mask!")
    return mask


def vlm_response_prompt_after_apply_instruction(url, 
                                                gpt_key, 
                                                image, 
                                                editing_prompt,
                                                category,
                                                device):
                                                
    try:

        base64_image = encode_image(image)  
        if category=="Removal":
            messages = create_editing_remove(editing_prompt, base64_image)
        elif category=="Local":
            messages = create_editing_replace(editing_prompt, base64_image)
        elif category=="Addition":
            messages = create_editing_add(editing_prompt, base64_image)
        else:
            messages = create_apply_editing_messages_gpt4o(editing_prompt, base64_image)
        response_str = run_azure_openai_inference(url, gpt_key, messages)
    except Exception as e:
        raise gr.Error("Please select the correct VLM model and input the correct API Key first!2")
    return response_str

def detect_object(url, gpt_key, image):
    try:

        base64_image = encode_image(image)  
        messages = detect_main_object_messages_gpt4o(base64_image)
        response_str = run_azure_openai_inference(url, gpt_key, messages)
    except Exception as e:
        raise gr.Error("Please select the correct VLM model and input the correct API Key first!2")
    return response_str