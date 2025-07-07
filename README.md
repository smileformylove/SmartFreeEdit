<div align="center">


# SmartFreeEdit: Mask-Free Spatial-Aware Image Editing with Complex Instruction Understanding

[Qianqian Sun](https://github.com/Decemberease)<sup>1*</sup>, [Jixiang Luo](https://github.com/smileformylove)<sup>2†</sup>, Dell Zhang<sup>2</sup>,  Xuelong Li<sup>2</sup>

<sup>1</sup>  The University of Hongkong,  <sup>2</sup> Institute of Artificial Intelligence (TeleAI)

<a href='https://smartfreeedit.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![arXiv](https://img.shields.io/badge/arXiv-2504.12704-b31b1b.svg)](https://arxiv.org/abs/2504.12704)
[![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-orange.svg)](https://huggingface.co/SUNQ111/SmartFreeEdit/tree/main) 


</div>

<p>
We propose SmartFreeEdit to address the challenge of reasoning instructions and segmentations  in image editing, thereby enhancing the practicality of AI editing. Our method effectively handles some semantic editing operations, including adding, removing, changing objects, background changing and global editing.
</p>



<p align="center">
<img src="assets/examples.png" width="1080px"/>
</p>


# ⏱️ Update News

- [2025.7.07] Webpage has been released!
- [2025.7.05] Our paper has been accpeted by ACMMM'2025 and Code for image editing is released!
- [2025.4.17] Our paper has been released on [arxiv Papers](https://arxiv.org/abs/2504.12704) and is currently under review for ACM Multimedia 2025 (ACMMM 2025).


# 📖 Pipeline

<p align="center">
<img src="assets/pipeline.png" width="1080px"/>
<p>
Our SmartFreeEdit consists of three key components: 1) An MLLM-driven Promptist that decomposes instructions into Editing Objects, Category, and Target Prompt. 2) Reasoning segmentation converts the prompt into an inference query and generates reasoning masks. 3) An Inpainting-based Image Editor using the hypergraph computation module to enhance global image structure understanding for more accurate edits.


# 🚀 Getting Started

## Environment Requirement 🌍


Clone the repo:

```
git clone https://github.com/smileformylove/SmartFreeEdit
```

We recommend you first use conda to create virtual environment, then run:

```
conda create -n smartfreeedit python=3.10.6 -y
conda activate smartfreeedit
python -m pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e .
```

Finally, you should install the remaining environment:
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

```

## Downloading Checkpoints

Checkpoints of SmartFreeEdit can be downloaded using the following command:

```
python SmartFreeEdit/download.py
```

## Running Gradio demo 

We provide a demo scripts for different hardware configurations. For users with server access and sufficient CPU/GPU memory ( >40/24 GB), we recommend you use:


```
export PYTHONPATH=.:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

python SmartFreeEdit/src/smartfreeedit_app.py
```

## Training
To train SmartFreeEdit, you need to download the BrushData [here](https://huggingface.co/random123123) and Checkpoints of BrushNet [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp=drive_link).

The data strcture should be like:

```
|-- data
    |-- BrushData
    |-- BrushDench
    |-- EditBench
    |-- ckpt
        |-- realisticVisionV60B1_v51VAE
            |-- model_index.json
            |-- vae
            |-- ...
        |-- segmentation_mask_brushnet_ckpt
        |-- segmentation_mask_brushnet_ckpt_sdxl_v0
        |-- random_mask_brushnet_ckpt
        |-- random_mask_brushnet_ckpt_sdxl_v0
        |-- ...
```

Train with segmentation mask using the script:

```
accelerate launch train/brushnet/train_brushnet.py \
--pretrained_model_name_or_path data/base_model/realisticVisionV60B1_v51VAE \
--output_dir examples/hyper \
--train_data_dir data/BrushData \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 8 \
--tracker_project_name brushnet \
--report_to tensorboard \
--validation_steps 300
```

You can inference with the script:

```
python train/brushnet/test_brushnet.py
```

You can evaluate using the script:
```
python train/brushnet/evaluate_brushnet.py \
--brushnet_ckpt_path data/ckpt/segmentation_mask_brushnet_ckpt \
--image_save_path runs/evaluation_result/BrushBench/brushnet_segmask/inside \
--mapping_file data/BrushBench/mapping_file.json \
--base_dir data/BrushBench \
--mask_key inpainting_mask
```
## Inference

Please download [Reason-Edit evaluation benchmark](https://drive.google.com/drive/folders/1QGmye23P3vzBBXjVj2BuE7K3n8gaWbyQ) from [SmartEdit](https://github.com/TencentARC/SmartEdit/tree/main) and put it in file dataset.

Use the script to inference on understanding and reasoning scenes:
```

python test/ReasonEdit_test.py --save_dir /SmartFreeEdit/edited_images --ReasonEdit_benchmark_dir /dataset/Reasonedit

```

# 🖋️ Citation

If you find our work helpful, please **star ⭐**this repo and **cite 📑** our paper. Thanks for your support!

```
@article{sun2025smartfreeedit,
  title={SmartFreeEdit: Mask-Free Spatial-Aware Image Editing with Complex Instruction Understanding},
  author={Sun, Qianqian and Luo, Jixiang and Zhang, Dell and Li, Xuelong},
  journal={arXiv preprint arXiv:2504.12704},
  year={2025}
}
```



# 📧 Contact

This repository is currently under active development and restructuring. The codebase is being optimized for better stability and reproducibility. While we strive to maintain code quality, you may encounter temporary issues during this transition period. For any questions or technical discussions, feel free to open an issue or contact us via email at sqq134050@163.com.

# 👍🏻 Acknowledgements
Our code is modified based on [BrushNet](https://github.com/TencentARC/BrushNet), thanks to all the contributors!
