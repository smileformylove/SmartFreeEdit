import argparse
import os
import numpy as np
from PIL import Image
from matrics_calculator import MetricsCalculator
import random
import torch
import ipdb

from huggingface_hub import snapshot_download
import gradio as gr
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
from SmartFreeEdit.utils.utils import load_grounding_dino_model, get_grounding_output
from groundingdino.datasets import transforms as T

import glob

VLM_MODEL_NAMES = list(vlms_template.keys())
DEFAULT_VLM_MODEL_NAME = "GPT4-o (Highly Recommended)"
BASE_MODELS = list(base_models_template.keys())
DEFAULT_BASE_MODEL = "realisticVision (Default)"

ASPECT_RATIO_LABELS = list(aspect_ratios)
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_LABELS[1]

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
    change boxes to [top-left x coordinate, top-left y coordinate, box width, box height] format.
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
    GPT4o_KEY, api_version, 
    end_point, engine
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
                        lisa_model, tokenizer, "cuda", boxs)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default="test/results",
        help="Directory to save the results"
    )
    parser.add_argument(
        '--ReasonEdit_benchmark_dir',
        type=str,
        default="",
        help="Directory containing the ReasonEdit benchmark dataset"
    )

    parser.add_argument('--result_path', type=str, default="test/evaluation_result.csv")
    parser.add_argument('--device', type=str, default="cuda", help="GPU device")
    parser.add_argument('--api_key', type=str, default="", help="GPT API key")
    parser.add_argument('--api_version', type=str, default="2024-08-01-preview", help="API version")
    parser.add_argument('--end_point', type=str, default="https://06.openai.azure.com/", help="API endpoint")
    parser.add_argument('--engine', type=str, default="4o", help="API engine")
    args = parser.parse_args()
    torch_dtype = torch.float16

    try:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except:
        device = "cpu"

    torch_dtype = torch.float16
    # download hf models
    SmartFreeEdit_path = "/models"
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
    result_path=args.result_path
    metrics_calculator=MetricsCalculator(args.device)

    ReasonEdit_benchmark_dir = args.ReasonEdit_benchmark_dir
    benchmark_understanding_scenes_dir = []
    for root, dirs, files in os.walk(ReasonEdit_benchmark_dir):
        for dir in dirs:
            if dir.endswith('1-Left-Right') or dir.endswith('2-Relative-Size') or dir.endswith('3-Mirror') or dir.endswith('4-Color') or dir.endswith('5-Multiple-Objects'):
                print(os.path.join(root, dir))
                sub_path = os.path.join(root, dir)
                benchmark_understanding_scenes_dir.append(sub_path)

    for sub_dir in benchmark_understanding_scenes_dir:
        total_save_path = args.save_dir
        if sub_dir.endswith("1-Left-Right") == True:
            test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            with open(sub_dir + "/Left_Right_text.txt", 'r') as f:
                prompt = f.readlines()
            final_save_dir = total_save_path + "/LeftRight_1"

        if sub_dir.endswith("2-Relative-Size") == True:
            test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            with open(sub_dir + "/Size_text.txt", 'r') as f:
                prompt = f.readlines()
            final_save_dir = total_save_path + "/RelativeSize_2"
        if sub_dir.endswith("3-Mirror") == True:
            test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            with open(sub_dir + "/Mirror_text.txt", 'r') as f:
                prompt = f.readlines()
            final_save_dir = total_save_path + "/Mirror_3"
        if sub_dir.endswith("4-Color") == True:
            test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            with open(sub_dir + "/Color_text.txt", 'r') as f:
                prompt = f.readlines()
            final_save_dir = total_save_path + "/Color_4"
        if sub_dir.endswith("5-Multiple-Objects") == True:
            test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            with open(sub_dir + "/MultipleObjects_text.txt", 'r') as f:
                prompt = f.readlines()
            final_save_dir = total_save_path + "/MultipleObjects_5"


        for idx, img_path in enumerate(test_img_list):
            text_prompt = prompt[idx].strip()
            editing_prompt = text_prompt.split("CLIP: ")[0].rstrip()
            src_image = Image.open(img_path)
            input_image = {
            "background": src_image,
            "layers": [src_image.convert("RGBA")]
                }

            os.makedirs(final_save_dir, exist_ok=True)
            result_image_path = os.path.join(final_save_dir, f"{idx+1}_result.png")
            if os.path.exists(result_image_path):
                print(f"Result image {idx+1} already exists. Skipping...")
                continue
            try:
                result_image, mask_images, masked_images, out_prompt, _ = process(input_image, 
                                                                                original_image=None, 
                                                                                original_mask=None, 
                                                                                prompt=text_prompt, 
                                                                                negative_prompt="", 
                                                                                control_strength=1.0, 
                                                                                seed=12345, 
                                                                                randomize_seed=False, 
                                                                                guidance_scale=7.5, 
                                                                                num_inference_steps=50,
                                                                                num_samples=1,
                                                                                blending=True,
                                                                                category=None,
                                                                                target_prompt="",
                                                                                resize_default=False,
                                                                                aspect_ratio_name=DEFAULT_ASPECT_RATIO,
                                                                                GPT4o_KEY=args.api_key, 
                                                                                api_version=args.api_version, 
                                                                                end_point=args.end_point, 
                                                                                engine=args.engine)
                print(f"Result image: {result_image}")
                selected_image = result_image[0]
                #AttributeError: 'list' object has no attribute 'save'
                selected_image.save(result_image_path)
            except Exception as e:
                ipdb.set_trace()
                print(f"Error processing image {idx}: {e}")
                continue
    test_dir = ReasonEdit_benchmark_dir + "/6-Reasoning"
    test_img_list = sorted(glob.glob(f'{test_dir}/*.png'))
    with open(test_dir + "/Reason_test.txt", 'r') as f:
            prompt = f.readlines()
    total_save_path = args.save_dir
    final_save_dir = total_save_path + "/Reason_6"
    os.makedirs(final_save_dir, exist_ok=True)
    for idx, img_path in enumerate(test_img_list):
        text_prompt = prompt[idx]
        editing_prompt = text_prompt.split("CLIP: ")[0].rstrip()
        src_image = Image.open(img_path)
        input_image = {
            "background": src_image,
            "layers": [src_image.convert("RGBA")]
        }
        result_image_path = os.path.join(final_save_dir, f"{idx+1}_result.png")
        if os.path.exists(result_image_path):
            continue
        try:
            result_image, mask_images, masked_images, out_prompt, _ = process(input_image, 
                                                                            original_image=None, 
                                                                            original_mask=None, 
                                                                            prompt=text_prompt, 
                                                                            negative_prompt="", 
                                                                            control_strength=1.0, 
                                                                            seed=12345, 
                                                                            randomize_seed=False, 
                                                                            guidance_scale=7.5, 
                                                                            num_inference_steps=50,
                                                                            num_samples=1,
                                                                            blending=True,
                                                                            category=None,
                                                                            target_prompt="",
                                                                            resize_default=False,
                                                                            aspect_ratio_name=DEFAULT_ASPECT_RATIO,
                                                                            GPT4o_KEY=args.api_key, 
                                                                            api_version=args.api_version, 
                                                                            end_point=args.end_point, 
                                                                            engine=args.engine)
            print(f"Result image: {result_image}")
            selected_image = result_image[0]
            selected_image.save(result_image_path)
        except Exception as e:
            ipdb.set_trace()
            print(f"Error processing image {idx+1}: {e}")
            continue