from PIL import Image, ImageEnhance
from diffusers.image_processor  import VaeImageProcessor

import numpy as np
import cv2



def SmartFreeEdit_Pipeline(pipe, 
                    prompts,
                    mask_np,
                    original_image, 
                    generator,
                    num_inference_steps,
                    guidance_scale,
                    control_strength,
                    negative_prompt,
                    num_samples,
                    blending):
    if mask_np.ndim != 3:
        mask_np = mask_np[:, :, np.newaxis]
    
    mask_np = mask_np / 255
    height, width = mask_np.shape[0], mask_np.shape[1]
    ## resize the mask and original image to the same size which is divisible by vae_scale_factor
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True)
    height_new, width_new = image_processor.get_default_height_width(original_image, height, width)
    mask_np = cv2.resize(mask_np, (width_new, height_new))[:,:,np.newaxis]
    dilation_size = int(max(height_new, width_new) * 0.05)  # 计算膨胀的像素数
    kernel = np.ones((dilation_size, dilation_size), np.uint8)  # 定义膨胀核
    mask_dilated = cv2.dilate((mask_np * 255).astype(np.uint8), kernel, iterations=1)
    mask_dilated = mask_dilated.astype(np.float32) / 255.0
    mask_blurred = cv2.GaussianBlur(mask_dilated*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:, :, np.newaxis]
    original_image = cv2.resize(original_image, (width_new, height_new))

    init_image = original_image * (1 - mask_np)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np.repeat(3, -1) * 255).astype(np.uint8)).convert("RGB")

    brushnet_conditioning_scale = float(control_strength)
    
    images = pipe(
        [prompts] * num_samples, 
        init_image, 
        mask_image, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        negative_prompt=[negative_prompt]*num_samples,
        height=height_new,
        width=width_new,
    ).images
    ## convert to vae shape format, must be divisible by 8
    original_image_pil = Image.fromarray(original_image).convert("RGB")
    init_image_np = np.array(image_processor.preprocess(original_image_pil, height=height_new, width=width_new).squeeze())
    init_image_np = ((init_image_np.transpose(1,2,0) + 1.) / 2.) * 255
    init_image_np = init_image_np.astype(np.uint8)
    if blending:
        mask_blurred = mask_blurred * 0.5 + 0.5
        image_all = []
        for image_i in images:
            image_np = np.array(image_i)
            ## blending
            image_pasted = init_image_np * (1 - mask_blurred) + mask_blurred * image_np
            image_pasted = image_pasted.astype(np.uint8)
            image = Image.fromarray(image_pasted)
            image_all.append(image)
    else:
        image_all = images


    return image_all, mask_image, mask_np, init_image_np


