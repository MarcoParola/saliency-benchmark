import torch.nn as nn
from PIL import Image
from gradio_client import Client


def handle_smallcap_result(output_smallcaps):
    prompt = output_smallcaps.split("Retrieved captions:")

    prompt = [caption.replace('\n', ' ') for caption in prompt]
    #print(prompt)

    caption = ', '.join(prompt)
    return caption


class SmallCapModule(nn.Module):
    def __init__(self):
        super(SmallCapModule, self).__init__()
        self.client = Client("https://ritaparadaramos-smallcapdemo.hf.space/")

    def forward(self, path):
        result = self.client.predict(
            path,  # str (filepath or URL to image) in 'image' Image component
            api_name="/predict"
        )
        return result


if __name__ == '__main__':
    IMAGE_PATH = "../../../images/peopleAndFish.jpg"

    img = Image.open(IMAGE_PATH)
    small_module = SmallCapModule()
    print(small_module(IMAGE_PATH))
