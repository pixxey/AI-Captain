import os
import requests
import base64
import time
import subprocess
import random
from datetime import datetime

AUTOMATIC1111_REPO = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui'
LOCALHOST_URL = "http://127.0.0.1:7860"
API_URL = f"{LOCALHOST_URL}/sdapi/v1/txt2img"
CHECK_INTERVAL = 5
OUTPUT_BASE_DIR = "outputs"

# List of 20 random prompts
PROMPTS = [
    "A futuristic cityscape",
    "A serene forest with a flowing river",
    "A bustling marketplace in an ancient city",
    "A mystical castle floating in the sky",
    "A vibrant underwater coral reef",
    "A spaceship landing on an alien planet",
    "A tranquil beach at sunset",
    "A snowy mountain range",
    "A lively carnival with colorful lights",
    "A peaceful meadow with wildflowers",
    "A dark and eerie haunted house",
    "A futuristic robot in a city",
    "A dragon flying over a village",
    "A majestic waterfall in a dense jungle",
    "A sci-fi laboratory with advanced technology",
    "A beautiful galaxy with swirling stars",
    "A knight in shining armor",
    "A futuristic skyline at night",
    "A quaint village in the countryside",
    "A magical forest with glowing mushrooms"
]

def check_installation():
    if os.path.exists('stable-diffusion-webui'):
        print("Automatic1111 found. Not cloning.")
        return True
    else:
        print("Automatic1111 not found. Now cloning.")
        return False

def install_automatic1111():
    print("Cloning Automatic1111 repository...")
    try:
        subprocess.run(['git', 'clone', AUTOMATIC1111_REPO], check=True)
        print("Cloning completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during cloning: {e}")

def check_if_running():
    try:
        response = requests.get(LOCALHOST_URL)
        if response.status_code == 200:
            print("Automatic1111 is running, skipping start.")
            return True
    except requests.ConnectionError:
        return False

def wait_for_server():
    print("Waiting for the server to be ready...")
    while not check_if_running():
        time.sleep(CHECK_INTERVAL)
    print("Server is ready.")

def launch_automatic1111():
    os.chdir('stable-diffusion-webui')
    print("Starting Automatic1111 server")
    subprocess.Popen(['python3', 'launch.py', '--no-half', '--api'])
    wait_for_server()

def save_image(image_data, output_dir, index):
    image_filename = os.path.join(output_dir, f"output_{index}.png")
    with open(image_filename, "wb") as file:
        file.write(image_data)
    return image_filename

def generate_images(prompt, num_images=1, steps=50, batch_number=1):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_BASE_DIR, f"output_{current_time}_batch_{batch_number}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    payload = {
        "prompt": prompt,
        "steps": steps,
        "batch_size": num_images
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        end_time = time.time()
    except requests.RequestException as e:
        print(f"Error generating images: {e}")
        return

    inference_time = (end_time - start_time)
    iterations_per_second = steps / inference_time
    image_filenames = []

    if 'images' in result:
        for i, image in enumerate(result['images']):
            image_data = base64.b64decode(image)
            image_filenames.append(save_image(image_data, output_dir, i))
    else:
        print("No images found in the response")

    save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames, batch_number)

def save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames, batch_number):
    metrics_filename = os.path.join(output_dir, f"metrics_batch_{batch_number}.md")
    metrics_content = (
        f"\n# Generation Metrics (Batch {batch_number})\n"
        f"- **Prompt**: {payload['prompt']}\n"
        f"- **Steps**: {payload['steps']}\n"
        f"- **Batch Size**: {payload['batch_size']}\n"
        f"- **Inference Time**: {inference_time:.2f} seconds\n"
        f"- **Iterations per Second**: {iterations_per_second:.2f}\n"
        f"- **Generated Images**:\n"
    )
    for filename in image_filenames:
        metrics_content += f"  - {filename}\n"

    with open(metrics_filename, "w") as metrics_file:
        metrics_file.write(metrics_content)

    # Print metrics to console
    print(metrics_content)
    print(f"Metrics saved to {metrics_filename}")

if __name__ == "__main__":
    if not check_installation():
        install_automatic1111()

    server_running = check_if_running()
    if not server_running:
        launch_automatic1111()

    # Select a random prompt from the list
    selected_prompt = random.choice(PROMPTS)
    print(f"Selected prompt: {selected_prompt}")

    # Generate images using the selected prompt in three batches
    generate_images(selected_prompt, num_images=1, batch_number=1)
    generate_images(selected_prompt, num_images=5, batch_number=2)
    generate_images(selected_prompt, num_images=10, batch_number=3)
