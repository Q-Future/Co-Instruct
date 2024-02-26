<div align="center">
  <h1>Towards Open-ended Visual Quality Comparison</h1> 
    
<div style="width: 50%; text-align: center; margin:auto;">
      <img style="width: 50%" src="teaser.png">
</div> 
</div>

_An study on open-ended multi-image quality comparison: a dataset, a model and a benchmark._

## Demo

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.50.0/gradio.js"
></script>

<gradio-app src="https://q-future-co-instruct.hf.space"></gradio-app>

## News

[Feb 24, 2024] A technical report for the data and model is coming soon.

## Load Model

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("q-future/co-instruct", 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             attn_implementation="eager", 
                                             device_map={"":"cuda:0"})
```

## Chat

```python
import requests
from PIL import Image


### Single Image
prompt = "USER: The image: <|image|> Which happens in this image: motion-blur, over-exposure, or under-exposure? ASSISTANT:"
url = "https://raw.githubusercontent.com/Q-Future/Q-Align/main/fig/singapore_flyer.jpg"
image = Image.open(requests.get(url,stream=True).raw)
model.chat(prompt, [image], max_new_tokens=200)

## Motion blur

### Double Image Comparison
prompt_cmp = "USER: The first image: <|image|>\nThe second image: <|image|>Which image has better quality, and why? ASSISTANT:"
url = "https://raw.githubusercontent.com/Q-Future/Q-Align/main/fig/boy_colorful.jpg"
image_2 = Image.open(requests.get(url,stream=True).raw)
model.chat(prompt_cmp, [image, image_2], max_new_tokens=200)

## The second image has better quality. The description indicates that the image has accurate exposure, precise focus, clear details, rich colors, and sufficient lighting. Additionally, the texture details are clear, and the composition is centered. In comparison, the first image has good clarity and rich texture details, but the lighting is slightly weak, which can affect the overall quality of the image. Therefore, the second image is of higher quality due to its accurate exposure, precise focus, clear details, rich colors, sufficient lighting, and centered composition.

```
