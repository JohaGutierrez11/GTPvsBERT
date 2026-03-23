from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")

image = pipe("una zapatilla deportiva futurista en una ciudad moderna").images[0]

image.save("demo.png")