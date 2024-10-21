def handle_llama_result(output_llama):
    start_index = output_llama.rfind("<|end_header_id|>") + len("<|end_header_id|>")
    caption = output_llama[start_index:].split("<|eot_id|>")[0].strip()
    return caption


def handle_smallcap_result(output_smallcaps):
    prompt = output_smallcaps.split("Retrieved captions:")

    prompt = [caption.replace('\n', ' ') for caption in prompt]
    #print(prompt)

    caption = ', '.join(prompt)
    return caption
