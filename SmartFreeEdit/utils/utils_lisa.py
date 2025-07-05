from enum import Enum
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch
import cv2
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor, BitsAndBytesConfig
import torch.nn.functional as F
from datetime import datetime
import argparse
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def load_lisa_model(version, precision="fp16", load_in_8bit=True, load_in_4bit=False, vision_tower="openai/clip-vit-large-patch14", local_rank=0):
    # Initialize tokenizer
    from SmartFreeEdit.model.LISA import LISAForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=512,  # You can adjust this according to your model
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token  # Ensure pad token is set to unk token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Set the appropriate precision
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    # Set quantization options for 4-bit or 8-bit
    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    # Load the LISA model

    model = LISAForCausalLM.from_pretrained(
        version, 
        low_cpu_mem_usage=True, 
        vision_tower=vision_tower, 
        seg_token_idx=seg_token_idx, 
        use_cache=True,
        **kwargs
    )
    print("Model loaded successfully.")
    # Set EOS, BOS, and PAD tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Load the model to the correct device

    # Adjust precision as per input argument
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif precision == "fp16" and (not load_in_4bit) and (not load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None


        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()

    return model, tokenizer
def run_grounded_sam(input_image, 
                     text_prompt, 
                     task_type, 
                     model,
                     tokenizer,
                     device=None):
    from SmartFreeEdit.model.llava.mm_utils import tokenizer_image_token
    from SmartFreeEdit.model.segment_anything.utils.transforms import ResizeLongestSide
    from SmartFreeEdit.model.llava import conversation as conversation_lib
    def parse_args():
        parser = argparse.ArgumentParser(description="LISA chat")
        parser.add_argument("--version", default="/data2/sqq/models/LISA-7B-v1-explanatory")
        parser.add_argument("--vis_save_path", default="./vis_output", type=str)
        parser.add_argument(
            "--precision",
            default="fp16",
            type=str,
            choices=["fp32", "bf16", "fp16"],
            help="precision for inference",
        )
        parser.add_argument("--image_size", default=1024, type=int, help="image size")
        parser.add_argument("--model_max_length", default=512, type=int)
        parser.add_argument("--lora_r", default=8, type=int)
        parser.add_argument(
            "--vision-tower", default="openai/clip-vit-large-patch14", type=str
        )
        parser.add_argument("--local-rank", default=0, type=int, help="node rank")
        parser.add_argument("--load_in_8bit", action="store_true", default=True)
        parser.add_argument("--load_in_4bit", action="store_true", default=False)
        parser.add_argument("--use_mm_start_end", action="store_true", default=True)
        parser.add_argument(
            "--conv_type",
            default="llava_v1",
            type=str,
            choices=["llava_v1", "llava_llama_2"],
        )
        parser.add_argument('--save_dir', type=str, default=None, help='Directory to save edited images')
        parser.add_argument('--ReasonEdit_benchmark_dir', type=str, default=None, help='Directory for benchmark models')
        parser.add_argument('--api_key', type=str, default=None, help='API key for VLM model')

        return parser.parse_args()
    args = parse_args()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device)
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    image_np = input_image['image']
    # Ensure the image is in RGB format
    if image_np.shape[2] == 4:  # If the image has an alpha channel
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif image_np.shape[2] == 1:  # If the image is grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    original_size_list = [image_np.shape[:2]]

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    
    if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()
    transform = ResizeLongestSide(args.image_size)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    output_ids, pred_masks = model.evaluate(
        image_clip, image, input_ids, resize_list, original_size_list,
        max_new_tokens=512, tokenizer=tokenizer
    )
    # Filter and return the final mask based on thresholds
    final_mask = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            print(f"No mask found for index {i}")
            continue
        
        # Apply threshold and IOU filtering logic
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0  # Apply threshold for bounding box confidence
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # You can add logic here for handling IOU threshold or other filtering conditions
        save_path = f"./vis_output/{timestamp}_mask_{i}.jpg"
        cv2.imwrite(save_path, pred_mask * 255)
        #final_mask = torch.tensor(pred_mask).to(device)
        # print(f"Final mask shape: {pred_mask.shape}")
        # print(f"Final mask sum: {pred_mask.sum()}")

    if pred_mask is None:
        raise ValueError("No valid mask found for the given input.")

    return pred_mask * 255
