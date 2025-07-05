##!/usr/bin/python3
# -*- coding: utf-8 -*-
import os, random, sys
import numpy as np
import requests
import torch
import whisper
import gradio as gr
from PIL import Image
from pydub import AudioSegment
from huggingface_hub import snapshot_download
from scipy.ndimage import binary_dilation, binary_erosion
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from SmartFreeEdit.utils.utils_lisa import load_lisa_model

from SmartFreeEdit.src.vlm_pipeline import (
    vlm_response_editing_type,
    vlm_response_object_wait_for_edit, 
    vlm_response_mask, 
    vlm_response_prompt_after_apply_instruction,
    detect_object
)
from SmartFreeEdit.src.smartfreeedit_all_pipeline import SmartFreeEdit_Pipeline
from SmartFreeEdit.utils.utils import load_grounding_dino_model

from SmartFreeEdit.src.vlm_template import vlms_template
from SmartFreeEdit.src.base_model_template import base_models_template
from SmartFreeEdit.src.aspect_ratio_template import aspect_ratios
from openai import OpenAI
from SmartFreeEdit.utils.utils import load_grounding_dino_model, get_grounding_output
from groundingdino.datasets import transforms as T
import json
from openai import AzureOpenAI


VLM_MODEL_NAMES = list(vlms_template.keys())
DEFAULT_VLM_MODEL_NAME = "GPT4-o (Highly Recommended)"
BASE_MODELS = list(base_models_template.keys())
DEFAULT_BASE_MODEL = "realisticVision (Default)"

ASPECT_RATIO_LABELS = list(aspect_ratios)
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_LABELS[0]
## init device
try:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
except:
    device = "cpu"

torch_dtype = torch.float16
# download hf models
SmartFreeEdit_path = "models/"
if not os.path.exists(SmartFreeEdit_path):
    SmartFreeEdit_path = snapshot_download(
        repo_id="TeleAI/SmartFreeEdit",
        local_dir=SmartFreeEdit_path,
        token=os.getenv("HF_TOKEN"),
    )

## init default VLM
vlm_type, vlm_local_path, vlm_processor, vlm_model = vlms_template[DEFAULT_VLM_MODEL_NAME]
## init base model
base_model_path = os.path.join(SmartFreeEdit_path, "base_model/realisticVisionV60B1_v51VAE")
brushnet_path = os.path.join(SmartFreeEdit_path, "checkpoint-100000/brushnet")
lisa_path = os.path.join(SmartFreeEdit_path, "LISA-7B-v1-explanatory")

# input brushnetX ckpt path
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()


## init groundingdino_model
groundingdino_path = os.path.join(SmartFreeEdit_path, "grounding_dino/groundingdino_swint_ogc.pth")
config_file = 'SmartFreeEdit/utils/GroundingDINO_SwinT_OGC.py'
groundingdino_model = load_grounding_dino_model(config_file, groundingdino_path, device="cuda")

## init lisa_model
lisa_model, tokenizer = load_lisa_model(
        version=lisa_path,
        precision="fp16",
        load_in_8bit=True,
        load_in_4bit=False,
        vision_tower="openai/clip-vit-large-patch14",
        local_rank=0
    )


## Ordinary function
def resize(image: Image.Image, 
                    target_width: int, 
                    target_height: int) -> Image.Image:
    """
    Crops and resizes an image while preserving the aspect ratio.

    Args:
        image (Image.Image): Input PIL image to be cropped and resized.
        target_width (int): Target width of the output image.
        target_height (int): Target height of the output image.

    Returns:
        Image.Image: Cropped and resized image.
    """
    # Original dimensions
    resized_image = image.resize((target_width, target_height), Image.NEAREST)
    return resized_image

def random_mask_func(mask, dilation_type='square', dilation_size=20):
    # Randomly select the size of dilation
    binary_mask = mask.squeeze()>0

    if dilation_type == 'square_dilation':
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        dilated_mask = binary_dilation(binary_mask, structure=structure)
    elif dilation_type == 'square_erosion':
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        dilated_mask = binary_erosion(binary_mask, structure=structure)
    elif dilation_type == 'bounding_box':
        # find the most left top and left bottom point
        rows, cols = np.where(binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return mask  # return original mask if no valid points

        min_row = np.min(rows)
        max_row = np.max(rows)
        min_col = np.min(cols)
        max_col = np.max(cols)

        # create a bounding box
        dilated_mask = np.zeros_like(binary_mask, dtype=bool)
        dilated_mask[min_row:max_row + 1, min_col:max_col + 1] = True

    elif dilation_type == 'bounding_ellipse':
        # find the most left top and left bottom point
        rows, cols = np.where(binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return mask  # return original mask if no valid points

        min_row = np.min(rows)
        max_row = np.max(rows)
        min_col = np.min(cols)
        max_col = np.max(cols)

        # calculate the center and axis length of the ellipse
        center = ((min_col + max_col) // 2, (min_row + max_row) // 2)
        a = (max_col - min_col) // 2  # half long axis
        b = (max_row - min_row) // 2  # half short axis

        # create a bounding ellipse
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        ellipse_mask = ((x - center[0])**2 / a**2 + (y - center[1])**2 / b**2) <= 1
        dilated_mask = np.zeros_like(binary_mask, dtype=bool)
        dilated_mask[ellipse_mask] = True
    else:
        ValueError("dilation_type must be 'square' or 'ellipse'")

    # use binary dilation
    dilated_mask =  np.uint8(dilated_mask[:,:,np.newaxis]) * 255
    return dilated_mask


## Gradio component function
def update_vlm_model(vlm_name):
    global vlm_model, vlm_processor
    if vlm_model is not None:
        del vlm_model
        torch.cuda.empty_cache()

    vlm_type, vlm_local_path, vlm_processor, vlm_model = vlms_template[vlm_name]
    
    ## we recommend using preload models, otherwise it will take a long time to download the model. you can edit the code via vlm_template.py
    if vlm_type == "openai":
        vlm_model = OpenAI(api_key='')
        vlm_processor = ""
    return "success"


def update_base_model(base_model_name):
    global pipe
    ## we recommend using preload models, otherwise it will take a long time to download the model. you can edit the code via base_model_template.py
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
    base_model_path, pipe = base_models_template[base_model_name]
    if pipe != "":
        pipe.to("cuda")
    else:
        if os.path.exists(base_model_path):
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
            )
            # pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
        else:
            raise gr.Error(f"The base model {base_model_name} does not exist")
    return "success"


def submit_GPT4o_KEY(api_key, api_version, end_point, engine):
    global vlm_model, vlm_processor
    if vlm_model is not None:
        del vlm_model
        torch.cuda.empty_cache()
    try:
        url = f"{end_point}/openai/deployments/{engine}/chat/completions?api-version={api_version}"

        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say this is a test"}
            ]
        })
        headers = {
            'Content-Type': 'application/json',
            'api-key': api_key,
        }

        response = requests.post(url, headers=headers, data=payload, timeout=20)
        response.raise_for_status()  
        response_str = response.json()['choices'][0]['message']['content']

  
        vlm_model = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=end_point
        )
        vlm_processor = ""  

        print("Success, " + response_str, "GPT4-o (Highly Recommended)")
        return vlm_model, vlm_processor, "Success, GPT4-o (Highly Recommended)"
    except Exception as e:
        return f"Invalid GPT4o API Configuration: {str(e)}", "GPT4-o (Highly Recommended)"
    
def transform_image(image):
    image_pil = Image.fromarray(image)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def convert_boxes_format(boxes, image_shape):
    """
    change boxes to [top-left x coordinate, top-left y coordinate, box width, box height]
    """
    converted_boxes = []
    height, width = image_shape[:2]
    for box in boxes:
        x_center, y_center, box_width, box_height = box.numpy()
        x_min = (x_center - box_width / 2) * width
        y_min = (y_center - box_height / 2) * height
        box_width = box_width * width
        box_height = box_height * height
        converted_boxes.append([x_min, y_min, box_width, box_height])
    return np.array(converted_boxes)

def add_response_mask(url,
                      gpt_key,
                      image,
                      groundingdino_model):
    detected_obj = detect_object(url, gpt_key, image)
    detected_obj_list = detected_obj.split(',')
    detected_obj_list = [obj.strip() for obj in detected_obj_list]
    transformed_image = transform_image(image)
    boxes_dict = {}
    for obj in detected_obj_list:
        reasoning_prompt = obj
        boxes, scores, phrases = get_grounding_output(
            groundingdino_model, transformed_image, obj, 0.5, 0.5
        )
        converted_boxes = convert_boxes_format(boxes, image.shape)
        boxes_dict[obj] = converted_boxes
    print(boxes_dict)
    return boxes_dict


    
def process(input_image, 
    original_image, 
    original_mask, 
    prompt, 
    negative_prompt, 
    control_strength, 
    seed, 
    randomize_seed, 
    guidance_scale, 
    num_inference_steps,
    num_samples,
    blending,
    category,
    target_prompt,
    resize_default,
    aspect_ratio_name,
    GPT4o_KEY, api_version , end_point, engine
    ):
    if original_image is None:
        if input_image is None:
            raise gr.Error('Please upload an image')
        else:
            image_pil = input_image["background"].convert("RGB")
            original_image = np.array(image_pil)
    if prompt is None or prompt == "":
        if target_prompt is None or target_prompt == "":
            raise gr.Error("Please input your instructions, e.g., remove the xxx")
    
    alpha_mask = input_image["layers"][0].split()[3]
    input_mask = np.asarray(alpha_mask)
    output_w, output_h = aspect_ratios[aspect_ratio_name]
    if output_w == "" or output_h == "":    
        output_h, output_w = original_image.shape[:2]

        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
            original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
            original_image = np.array(original_image)
            if input_mask is not None:
                input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
                input_mask = np.array(input_mask)
            if original_mask is not None:
                original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
                original_mask = np.array(original_mask)
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        else:
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
            pass 
    else:
        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
        gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
        original_image = np.array(original_image)
        if input_mask is not None:
            input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
            input_mask = np.array(input_mask)
        if original_mask is not None:
            original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
            original_mask = np.array(original_mask)

    
    API_KEY = GPT4o_KEY
    API_VERSION = api_version
    END_POINT = end_point
    ENGINE = engine
    ##这个版本使用我们的Azure OpenAI API
    API_KEY = '04992b6701a34d9297fe2f94611a2e22'
    API_VERSION = ''
    END_POINT = end_point
    ENGINE = engine
    url = f"{END_POINT}/openai/deployments/{ENGINE}/chat/completions?api-version={API_VERSION}"

    category = vlm_response_editing_type(url, API_KEY, original_image, prompt, "cuda")
    print(f'category: {category}')

    if original_mask is not None:
        original_mask = np.clip(original_mask, 0, 255).astype(np.uint8)
    else:
        try:
            object_wait_for_edit = vlm_response_object_wait_for_edit(
                                                url, 
                                                API_KEY, 
                                                original_image,
                                                category, 
                                                prompt,
                                                "cuda")
            print(f'object_wait_for_edit: {object_wait_for_edit}')
            if category == "Addition":
                boxs = add_response_mask(url, API_KEY, original_image, groundingdino_model)
                original_mask = vlm_response_mask(url, API_KEY, category, original_image, 
                        prompt, object_wait_for_edit, 
                        lisa_model, tokenizer, device, boxs)
            else:
                original_mask = vlm_response_mask(url, API_KEY, category, original_image,
                                                prompt, 
                                                object_wait_for_edit, 
                                                lisa_model,
                                                tokenizer,
                                                "cuda"
                                        ).astype(np.uint8)
        except Exception as e:
            raise gr.Error(f"{e}")

    if original_mask.ndim == 2:
        original_mask = original_mask[:,:,None]
    

    if target_prompt is not None and len(target_prompt) >= 1:
        prompt_after_apply_instruction = target_prompt
        
    else:
        try:
            prompt_after_apply_instruction = vlm_response_prompt_after_apply_instruction(
                                                                    url,
                                                                    API_KEY,
                                                                    original_image,
                                                                    prompt,
                                                                    category,
                                                                    "cuda")
        except Exception as e:
            raise gr.Error(f"{e}")

    generator = torch.Generator("cuda").manual_seed(random.randint(0, 2147483647) if randomize_seed else seed)


    with torch.autocast("cuda"):
        image, mask_image, mask_np, init_image_np = SmartFreeEdit_Pipeline(pipe, 
                                    prompt_after_apply_instruction,
                                    original_mask,
                                    original_image,
                                    generator,
                                    num_inference_steps,
                                    guidance_scale,
                                    control_strength,
                                    negative_prompt,
                                    num_samples,
                                    blending)
    original_image = np.array(init_image_np)
    masked_image = original_image * (1 - (mask_np>0))
    masked_image = masked_image.astype(np.uint8)
    masked_image = Image.fromarray(masked_image)
    gr.Info(f"Target Prompt: {prompt_after_apply_instruction}", duration=20)
    return image, [mask_image], [masked_image], prompt, ''


def generate_target_prompt( input_image, 
                            original_image, 
                            prompt,
                            gpt4o_key,
                            api_version,
                            end_point,
                            engine
                           ):
    # load example image
    if isinstance(original_image, str):
        original_image = input_image
    category = 'Localize'
    API_KEY = gpt4o_key
    API_VERSION = api_version
    END_POINT = end_point
    ENGINE = engine
    url = f"{END_POINT}/openai/deployments/{ENGINE}/chat/completions?api-version={API_VERSION}"
    try:
        prompt_after_apply_instruction = vlm_response_prompt_after_apply_instruction(
                                                            url, 
                                                            API_KEY, 
                                                            original_image,
                                                            prompt,
                                                            category,
                                                            "cuda")
    except Exception as e:
        raise gr.Error(f"{e}")
    
    return prompt_after_apply_instruction


def process_mask(input_image, 
    original_image, 
    prompt,
    resize_default,
    aspect_ratio_name):
    if original_image is None:
        raise gr.Error('Please upload the input image')
    if prompt is None:
        raise gr.Error("Please input your instructions, e.g., remove the xxx")

    ## load mask
    alpha_mask = input_image["layers"][0].split()[3]
    input_mask = np.array(alpha_mask)

    # load example image
    if isinstance(original_image, str):
        original_image = input_image["background"]

    if input_mask.max() == 0:
        category = vlm_response_editing_type(vlm_processor, vlm_model, original_image, prompt, "cuda")

        object_wait_for_edit = vlm_response_object_wait_for_edit(vlm_processor, 
                                                                vlm_model, 
                                                                original_image,
                                                                category, 
                                                                prompt,
                                                                "cuda")
        # original mask: h,w,1 [0, 255]
        original_mask = vlm_response_mask(
                                vlm_processor,
                                vlm_model,
                                category, 
                                original_image, 
                                prompt, 
                                object_wait_for_edit, 
                                groundingdino_model,
                                lisa_model,
                                tokenizer,
                                "cuda").astype(np.uint8)
    else:
        original_mask = input_mask.astype(np.uint8)
        category = None

    ## resize mask if needed
    output_w, output_h = aspect_ratios[aspect_ratio_name]
    if output_w == "" or output_h == "":    
        output_h, output_w = original_image.shape[:2]
        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
            original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
            original_image = np.array(original_image)
            if input_mask is not None:
                input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
                input_mask = np.array(input_mask)
            if original_mask is not None:
                original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
                original_mask = np.array(original_mask)
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        else:
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
            pass 
    else:
        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
        gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
        original_image = np.array(original_image)
        if input_mask is not None:
            input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
            input_mask = np.array(input_mask)
        if original_mask is not None:
            original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
            original_mask = np.array(original_mask)


    if original_mask.ndim == 2:
        original_mask = original_mask[:,:,None]

    mask_image = Image.fromarray(original_mask.squeeze().astype(np.uint8)).convert("RGB")

    masked_image = original_image * (1 - (original_mask>0))
    masked_image = masked_image.astype(np.uint8)
    masked_image = Image.fromarray(masked_image)

    return [masked_image], [mask_image], original_mask.astype(np.uint8), category


def process_random_mask(input_image, 
                         original_image, 
                         original_mask, 
                         resize_default, 
                         aspect_ratio_name, 
                         ):

    alpha_mask = input_image["layers"][0].split()[3]
    input_mask = np.asarray(alpha_mask)
    
    output_w, output_h = aspect_ratios[aspect_ratio_name]
    if output_w == "" or output_h == "":    
        output_h, output_w = original_image.shape[:2]
        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
            original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
            original_image = np.array(original_image)
            if input_mask is not None:
                input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
                input_mask = np.array(input_mask)
            if original_mask is not None:
                original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
                original_mask = np.array(original_mask)
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        else:
            gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
            pass 
    else:
        if resize_default:
            short_side = min(output_w, output_h)
            scale_ratio = 640 / short_side
            output_w = int(output_w * scale_ratio)
            output_h = int(output_h * scale_ratio)
        gr.Info(f"Output aspect ratio: {output_w}:{output_h}")
        original_image = resize(Image.fromarray(original_image), target_width=int(output_w), target_height=int(output_h))
        original_image = np.array(original_image)
        if input_mask is not None:
            input_mask = resize(Image.fromarray(np.squeeze(input_mask)), target_width=int(output_w), target_height=int(output_h))
            input_mask = np.array(input_mask)
        if original_mask is not None:
            original_mask = resize(Image.fromarray(np.squeeze(original_mask)), target_width=int(output_w), target_height=int(output_h))
            original_mask = np.array(original_mask)


    if input_mask.max() == 0:
        original_mask = original_mask
    else:
        original_mask = input_mask
    
    if original_mask is None:
        raise gr.Error('Please generate mask first')

    if original_mask.ndim == 2:
        original_mask = original_mask[:,:,None]

    dilation_type = np.random.choice(['bounding_box', 'bounding_ellipse'])
    random_mask = random_mask_func(original_mask, dilation_type).squeeze()

    mask_image = Image.fromarray(random_mask.astype(np.uint8)).convert("RGB")

    masked_image = original_image * (1 - (random_mask[:,:,None]>0))
    masked_image = masked_image.astype(original_image.dtype)
    masked_image = Image.fromarray(masked_image)


    return [masked_image], [mask_image], random_mask[:,:,None].astype(np.uint8)

def init_img(base, 
             init_type, 
             prompt,
             aspect_ratio,
             example_change_times
             ):
    image_pil = base["background"].convert("RGB")
    original_image = np.array(image_pil)
    if max(original_image.shape[0], original_image.shape[1]) * 1.0 / min(original_image.shape[0], original_image.shape[1])>2.0:
        raise gr.Error('image aspect ratio cannot be larger than 2.0')
    
    if aspect_ratio not in ASPECT_RATIO_LABELS:
        aspect_ratio = "Custom resolution"
    return base, original_image, None, "", None, None, None, "", "", aspect_ratio, False, 0

def reset_func(input_image, 
               original_image, 
               original_mask, 
               prompt, 
               target_prompt, 
               gpt4o_key,
               api_version,
               end_point,
               engine):
    input_image = None
    original_image = None
    original_mask = None
    prompt = ''
    mask_gallery = []
    masked_gallery = []
    result_gallery = []
    target_prompt = ''
    audio_prompt = ''
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return input_image, original_image, original_mask, prompt, mask_gallery, masked_gallery, result_gallery, target_prompt, False, gpt4o_key, api_version, end_point, engine



def recognize_speech_with_whisper(audio_file):
    result = whisper.transcribe(audio_file)
    return result["text"]


def update_prompt_with_audio(audio_file, current_prompt):
    if not audio_file.endswith(".wav"):
        audio_file = convert_to_wav(audio_file)
    if audio_file is not None:
        recognized_text = recognize_speech_with_whisper(audio_file)
        if recognized_text:
            return recognized_text, f"🎤 recognized result: {recognized_text}"

    return current_prompt, "🎤 no speech input detected"


def generate_target(input_image, original_image, prompt, **kwargs):

    target_prompt = f"generated_target: {prompt}"
    return target_prompt

def run_process(audio_file, input_image, original_image, prompt, **kwargs):
    
    if audio_file is not None:
        prompt, _ = update_prompt_with_audio(audio_file, prompt)
    
  
    target_prompt = generate_target(input_image, original_image, prompt, **kwargs)
    
   
    return target_prompt
def convert_to_wav(audio_file):
    # 
    audio = AudioSegment.from_file(audio_file)
    # 
    wav_file = audio_file.replace(".mp3", ".wav")  
    audio.export(wav_file, format="wav")
    return wav_file
block = gr.Blocks(
        theme=gr.themes.Soft(
             radius_size=gr.themes.sizes.radius_none,
             text_size=gr.themes.sizes.text_md
         )
        )
with block as demo:


    original_image = gr.State(value=None)
    original_mask = gr.State(value=None)
    category = gr.State(value=None)
    status = gr.State(value=None)
    #invert_mask_state = gr.State(value=False)
    example_change_times = gr.State(value=0)


    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.ImageEditor( 
                    label="Input Image",
                    type="pil",
                    sources=["upload"],
                    interactive=True,
                    brush=gr.Brush(colors=["#FFFFFF"], default_size = 30, color_mode="fixed"),
                    layers = False,
                    height=960,
                    placeholder="Please click here or the icon below to upload the image.",
                    )

            prompt = gr.Textbox(label="📖 Instruction", placeholder="Please input your instruction.", value="",lines=1)
            audio_prompt = gr.Audio(label="🎤 Voice Input", type="filepath")
            
            
            vlm_model_dropdown = gr.Dropdown(label="VLM model", choices=VLM_MODEL_NAMES, value=DEFAULT_VLM_MODEL_NAME, interactive=True)
            run_button = gr.Button("⭐ Run")
            with gr.Group():    
                with gr.Column():
                    GPT4o_KEY = gr.Textbox(label="GPT4o API Key", placeholder="Please input your GPT4o API Key when use GPT4o VLM (highly recommended).", value="", lines=1)
                    API_VERSION = gr.Textbox(label="GPT4o API VERSION", placeholder="", value="", lines=1)
                    END_POINT = gr.Textbox(label="GPT4o END_POINT", placeholder="", value="", lines=1)
                    ENGINE = gr.Textbox(label="GPT4o ENGINE", placeholder="", value="", lines=1)
                    GPT4o_KEY_submit = gr.Button("Submit and Verify")

            
            aspect_ratio = gr.Dropdown(label="Output aspect ratio", choices=ASPECT_RATIO_LABELS, value=DEFAULT_ASPECT_RATIO)
            resize_default = gr.Checkbox(label="Short edge resize to 640px", value=True)

            with gr.Row():
                mask_button = gr.Button("Generate Mask")
                random_mask_button = gr.Button("Square/Circle Mask ")
            

            with gr.Row():
                generate_target_prompt_button = gr.Button("Generate Target Prompt")
                
            target_prompt = gr.Text(
                        label="Input Target Prompt",
                        max_lines=5,
                        placeholder="VLM-generated target prompt, you can first generate if and then modify it (optional)",
                        value='',
                        lines=2
                    )

            with gr.Accordion("Advanced Options", open=False, elem_id="accordion1"):
                base_model_dropdown = gr.Dropdown(label="Base model", choices=BASE_MODELS, value=DEFAULT_BASE_MODEL, interactive=True)
                negative_prompt = gr.Text(
                        label="Negative Prompt",
                        max_lines=5,
                        placeholder="Please input your negative prompt",
                        value='ugly, low quality',lines=1
                    )
                                    
                control_strength = gr.Slider(
                    label="Control Strength: ", show_label=True, minimum=0, maximum=1.1, value=1, step=0.01
                    )
                with gr.Group():
                    seed = gr.Slider(
                        label="Seed: ", minimum=0, maximum=2147483647, step=1, value=648464818
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                
                blending = gr.Checkbox(label="Blending mode", value=True)

                
                num_samples = gr.Slider(
                    label="Num samples", minimum=0, maximum=4, step=1, value=4
                )
                
                with gr.Group():
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=1,
                            maximum=12,
                            step=0.1,
                            value=7.5,
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=50,
                        )

            
        with gr.Column():
            with gr.Row():
                with gr.Tab(elem_classes="feedback", label="Masked Image"):
                    masked_gallery = gr.Gallery(label='Masked Image', show_label=True, elem_id="gallery", preview=True, height=360)
                with gr.Tab(elem_classes="feedback", label="Mask"):
                    mask_gallery = gr.Gallery(label='Mask', show_label=True, elem_id="gallery", preview=True, height=360)
                
           
            with gr.Tab(elem_classes="feedback", label="Output"):
                result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", preview=True, height=400)

            # target_prompt_output = gr.Text(label="Output Target Prompt", value="", lines=1, interactive=False)

            reset_button = gr.Button("Reset")

            init_type = gr.Textbox(label="Init Name", value="", visible=False)
            example_type = gr.Textbox(label="Example Name", value="", visible=False)

    input_image.upload(
        init_img,
        [input_image, init_type, prompt, aspect_ratio, example_change_times],
        [input_image, original_image, original_mask, prompt, mask_gallery, masked_gallery, result_gallery, target_prompt, init_type, aspect_ratio, resize_default, example_change_times]
    ) 
    #base, original_image, None, "", None, None, None, "", "", aspect_ratio, False, 0
    audio_prompt.change(
        fn=update_prompt_with_audio, 
        inputs=[audio_prompt, prompt], 
        outputs=[prompt]
    )


    ## vlm and base model dropdown
    vlm_model_dropdown.change(fn=update_vlm_model, inputs=[vlm_model_dropdown], outputs=[status])
    base_model_dropdown.change(fn=update_base_model, inputs=[base_model_dropdown], outputs=[status])


    GPT4o_KEY_submit.click(fn=submit_GPT4o_KEY, inputs=[GPT4o_KEY, API_VERSION , END_POINT, ENGINE], outputs=[GPT4o_KEY])
  


    ips=[input_image, original_image, original_mask, prompt, negative_prompt, 
         control_strength, seed, randomize_seed, guidance_scale, num_inference_steps,
         num_samples, blending, category, target_prompt, resize_default, aspect_ratio, GPT4o_KEY, API_VERSION , END_POINT, ENGINE
         ]

    ## run smartfreeedit
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, mask_gallery, masked_gallery, prompt, target_prompt])
     ## mask func
    mask_button.click(fn=process_mask, inputs=[input_image, original_image, prompt, resize_default, aspect_ratio], outputs=[masked_gallery, mask_gallery, original_mask, category])
    random_mask_button.click(fn=process_random_mask, inputs=[input_image, original_image, original_mask, resize_default, aspect_ratio], outputs=[masked_gallery, mask_gallery, original_mask])
    ## prompt func
    generate_target_prompt_button.click(fn=generate_target_prompt, inputs=[input_image, original_image, prompt, GPT4o_KEY, API_VERSION, END_POINT, ENGINE], outputs=[target_prompt])

    ## reset func
    reset_button.click(fn=reset_func, inputs=[input_image, original_image, original_mask, prompt, target_prompt, GPT4o_KEY, API_VERSION, END_POINT, ENGINE], outputs=[input_image, original_image, original_mask, prompt, mask_gallery, masked_gallery, result_gallery, target_prompt, resize_default, GPT4o_KEY, API_VERSION, END_POINT, ENGINE])
                                                                                                                                                                             
## if have a localhost access error, try to use the following code
demo.launch(server_name="0.0.0.0", server_port=8080)
#demo.launch()