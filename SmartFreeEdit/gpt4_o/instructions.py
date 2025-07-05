def create_editing_category_messages_gpt4o(editing_prompt):
    messages = [{
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will give you an editing instruction of the image. Please output which type of editing category it is in. You can choose from the following categories: \n\
                    1. Addition: Adding new objects within the images, e.g., add a bird \n\
                    2. Remove: Removing objects, e.g., remove the mask \n\
                    3. Local: Replace local parts of an object and later the object's attributes (e.g., make it smile) or alter an object's visual appearance without affecting its structure (e.g., change the cat to a dog) \n\
                    4. Global: Edit the entire image, e.g., let's see it in winter, Change the season from autumn to spring\n\
                    5. Background: Change the scene's background, e.g., have her walk on water, change the background to a beach, make the hedgehog in France, etc. \n\
                    Only output a single word, e.g., 'Addition'.\n\
                    6.Resize: Resizing the objects, e.g., minify the giraffe in the image"
                    
                },]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
            ]
            }]
    return messages
    

def create_ori_object_messages_gpt4o(editing_prompt):

    messages =  [
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "I will give you an editing instruction of the image. Please output the object needed to be edited. You only need to output the basic description of the object . \n \
                    For example, the editing instruction is 'Change the white cat to a black dog'. Then you need to output: 'a white cat'. Only output the new content. \n\
If the editing instruction is 'remove the tool that is used to cut fruits', then you need to output: 'a  tool that is used to cut fruits'. \n\
If the editing instruction is 'remove the object that can be used to have meals', then you need to output: ' the object that can be used to have meals'. \n\
Do not output anything else."
                    },]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": editing_prompt
                    }
                ]
                }
            ]
    return messages


def create_add_object_messages_gpt4o(boxs, editing_prompt, base64_image, height=640, width=640):
    ##boxs is dict
    boxs_str =  ""
    for key, value in boxs.items():
        boxs_str += f"{key}: {value} \n"
    size_str = f"The image size is height {height}px and width {width}px. The top - left corner is coordinate [0 , 0]. The bottom - right corner is coordinnate [{height} , {width}]. "

    messages = [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "I need to add an object to the image following the instruction: " + editing_prompt + ". " + size_str + " \n \
                    And I'll provide some information about the location of the main objects in the image: " + boxs_str + ". \n \
                    Can you give me a possible bounding box of the location for the added object and the box size can be larger, The size of the box must not be smaller than height/3 and width/3  \n\
                    Please output with the format of [top - left x coordinate , top - left y coordinate , box width , box height]. \n\
                    You should only output the bounding box position and nothing else. Please refer to the example below for the desired format.\n\
                    [Examples]\n \
                    [19, 101, 32, 153]\n  \
                    [54, 12, 242, 96]"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                        ]
                        }
                    ]
    return messages


def create_editing_remove(editing_prompt, base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image along with an editing instruction and the editing category is 'remove' . Please describe the new content that should be present in the image after applying the instruction. \n \
For example, if the original image content shows a grandmother wearing a mask and the instruction is 'remove the mask', your output should be: 'a grandmother'. \n \
If the original image has multiple animals or objects of the same kind, like three dogs in a grassland, the instruction is 'remove the right dog', your output should be: 'grasslandr' that not mention any removal. \n \
The output should not mention elements that have been changed or removed, such as 'mask' in this example. \n\
Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
   },]
            },      
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages
def create_editing_replace(editing_prompt, base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image along with an editing instruction and the editing category is 'replace' . Please describe the new content that should be present in the image after applying the instruction. \n \
                    For example, if the original image content shows a grandmother wearing a mask and the instruction is 'replace the mask with a hat', your output should be: 'a hat'. \n \
                    If the original image has multiple animals or objects of the same kind, like three dogs in a grassland, the instruction is 'replace the right dog with a cat', your output should be: 'a cat'. \n \
                    Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
            },]
            },      
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages

def create_editing_add(editing_prompt, base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image along with an editing instruction and the editing category is 'add' . Please describe the new content that should be present in the image after applying the instruction. \n \
                    For example, if the original image content shows a grandmother wearing a mask and the instruction is 'add a hat', your output should be: 'a hat'. \n \
                    Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
            },]
            },      
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages    


def create_apply_editing_messages_gpt4o(editing_prompt, base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image along with an editing instruction. Please describe the new content that should be present in the image after applying the instruction. \n \
                    For example, if the original image content shows a grandmother wearing a mask and the instruction is 'remove the mask', your output should be: 'a grandmother'. \n \
                    If the original image content shows a grandmother in garden and the instruction is 'let us see it in rain', your output should be: 'a grandmother in garden in rain'. \n \
                        The output should only include elements that remain in the image after the edit and should not mention elements that have been changed or removed, such as 'mask' in this example. \n\
                            Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
                },]
            },      
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages

def detect_main_object_messages_gpt4o(base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image. Please describe the main content that present in the imag. \n \
                    For example, if the original image content shows a grandmother wearing a mask, your output should be: 'grandmother, mask'. \n \
                            Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
                },]
            },      
            {
            "role": "user",
            "content": [
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages