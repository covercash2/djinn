"""Run some generations on my MBP"""
import torch
from diffusers import DiffusionPipeline


def run():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to(torch_dtype=torch.float32)

    run_generation(pipe, "hello world")


def run_generation(pipe: DiffusionPipeline, prompt: str):
    image = pipe(prompt=prompt).images[0]

    image.show()

    return


if __name__ == "__main__":
    run()
