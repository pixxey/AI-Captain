import time
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Load the pre-trained Stable Diffusion 2.1 model
model_id = "stabilityai/stable-diffusion-2-1"

if torch.cuda.is_available():
    device = torch.device ("cuda")
print(f"Using device: {device}")
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Define a simple prompt
prompt = "A fantasy landscape with mountains and rivers"

# Function to measure inference rate
def measure_inference_rate(prompt, pipe, num_iterations=10):
    start_time = time.time()
    for _ in range(num_iterations):
        with autocast("cuda"):
            image = pipe(prompt).images[0]
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    inference_rate = 1 / avg_time_per_inference
    
    return inference_rate, avg_time_per_inference

# Measure inference rate
inference_rate, avg_time_per_inference = measure_inference_rate(prompt, pipe)

print(f"Average time per inference: {avg_time_per_inference:.4f} seconds")
print(f"Inference rate: {inference_rate:.2f} inferences per second")
