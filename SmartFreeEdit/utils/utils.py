import numpy as np
import torch
import torchvision

from scipy import ndimage

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# SAM
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# GroundingDINO
from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_grounding_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def generate_caption(processor, blip_model, raw_image, device):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption



def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases



def run_grounded_sam1(input_image, 
                     text_prompt, 
                     task_type, 
                     box_threshold, 
                     text_threshold, 
                     iou_threshold, 
                     scribble_mode,
                     sam,
                     groundingdino_model,
                     sam_predictor=None,
                     sam_automask_generator=None,
                     device="cuda"):

    global blip_processor, blip_model, inpaint_pipeline

    # load image
    image = input_image["image"]
    scribble = input_image["mask"]
    size = image.size # w, h

    if sam_predictor is None:
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)

    image_pil = image.convert("RGB")
    image = np.array(image_pil)

    if task_type == 'scribble':
        sam_predictor.set_image(image)
        scribble = scribble.convert("RGB")
        scribble = np.array(scribble)
        scribble = scribble.transpose(2, 1, 0)[0]

        # 将连通域进行标记
        labeled_array, num_features = ndimage.label(scribble >= 255)

        # 计算每个连通域的质心
        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
        centers = np.array(centers)

        point_coords = torch.from_numpy(centers)
        point_coords = sam_predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
        point_coords = point_coords.unsqueeze(0).to(device)
        point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)
        if scribble_mode == 'split':
            point_coords = point_coords.permute(1, 0, 2)
            point_labels = point_labels.permute(1, 0)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            mask_input = None,
            boxes = None,
            multimask_output = False,
        )
    elif task_type == 'automask':
        masks = sam_automask_generator.generate(image)
    else:
        transformed_image = transform_image(image_pil)

        if task_type == 'automatic':
            # generate caption and tags
            # use Tag2Text can generate better captions
            # https://huggingface.co/spaces/xinyu1205/Tag2Text
            # but there are some bugs...
            blip_processor = blip_processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            blip_model = blip_model or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
            text_prompt = generate_caption(blip_processor, blip_model, image_pil, device)
            print(f"Caption: {text_prompt}")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
        )

        # process boxes
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()


        if task_type == 'seg' or task_type == 'inpainting' or task_type == 'automatic':
            sam_predictor.set_image(image)

            if task_type == 'automatic':
                # use NMS to handle overlapped boxes
                print(f"Before NMS: {boxes_filt.shape[0]} boxes")
                nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
                boxes_filt = boxes_filt[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                print(f"After NMS: {boxes_filt.shape[0]} boxes")
                print(f"Revise caption with number: {text_prompt}")

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            return masks
        else:
            print("task_type:{} error!".format(task_type))