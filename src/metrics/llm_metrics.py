import base64
from io import BytesIO

import torch
from gradio_client import Client
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, Blip2ForImageTextRetrieval
from PIL import Image
import requests


def vqa_score(img, caption):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    client = Client("zhiqiulin/VQAScore")
    result = client.predict(
        model_name="clip-flant5-xxl",
        images=image_base64,
        text=caption,
        api_name="/rank_images"
    )

    return result


def clip_score(device, img, captions):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_inputs = processor(text=[caption[:77] for caption in captions], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    image_inputs = torch.stack(
        [processor(images=img, return_tensors="pt").pixel_values.squeeze(0)]).to(device)
    with torch.no_grad():
        # Encode images and text
        image_features = model.get_image_features(pixel_values=image_inputs)
        text_features = model.get_text_features(input_ids=text_inputs)

        # Compute similarity
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        similarity = torch.matmul(text_features, image_features.T)

        # Convert similarity to probabilities
        probs = similarity.softmax(dim=-1).cpu().numpy()

    return similarity


def blip_score(device, img, captions):
    model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

    model.to(device)
    image = img

    texts = captions

    inputs = processor(images=image, text=texts, return_tensors="pt").to(device, torch.float16)
    itc_out = model(**inputs, use_image_text_matching_head=False)
    logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")

    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")

    return probs[0]
