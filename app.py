import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Generador de Imágenes Sin Restricciones"
)

iface.launch(share=True)