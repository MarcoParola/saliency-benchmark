import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForPreTraining, CLIPProcessor, CLIPModel
from gradio_client import Client, file
from io import BytesIO
import base64

from src.models.llm.llm_metrics import vqa_score, clip_score
from src.models.llm.smallcap_module import SmallCapModule
from src.models.llm.utils_llm import handle_llama_result, handle_smallcap_result


class Llama32Module(nn.Module):
    def __init__(self, text):
        super(Llama32Module, self).__init__()
        quant_config = BitsAndBytesConfig(load_in_4bit=True)

        self.text_prompt = text

        self.processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
        self.model = AutoModelForPreTraining.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
            low_cpu_mem_usage=True,
            quantization_config=quant_config
        )

    def forward(self, image):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.text_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=500)
        print(self.processor.decode(output[0]))
        return self.processor.decode(output[0])


if __name__ == '__main__':
    text = ("Can you produce a detailed and concise caption for this image? Without using a list, a single continuous "
            "sentence")

    IMAGE_PATH = "../../../images/peopleAndFish.jpg"

    img = Image.open(IMAGE_PATH)
    model = Llama32Module(text)
    output_llama = model(img)
    print("output llama:"+ output_llama)

    model = SmallCapModule()
    output_smallcap = model(IMAGE_PATH)
    print("output smallcap:"+ output_smallcap)

    caption_llama=handle_llama_result(output_llama)
    caption_smallcap=handle_smallcap_result(output_smallcap)

    captions=[caption_llama, caption_smallcap]

    #print("Result with VQAScore:" + vqa_score(img,caption))
    print("Result with CLIP:" + str(clip_score('cuda',img, captions)))
    print("Result with BLIP2:" + str(clip_score('cuda',img, captions)))

