import os
import requests
import base64
import time
import subprocess
from datetime import datetime

AUTOMATIC1111_REPO = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui'
LOCALHOST_URL = "http://127.0.0.1:7860"
API_URL = f"{LOCALHOST_URL}/sdapi/v1/txt2img"
CHECK_INTERVAL = 5
OUTPUT_BASE_DIR = "outputs"

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

def generate_images(prompt, num_images=10, steps=50):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_BASE_DIR, f"output_{current_time}")

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

    inference_time = end_time - start_time
    iterations_per_second = steps / inference_time
    image_filenames = []

    if 'images' in result:
        for i, image in enumerate(result['images']):
            image_data = base64.b64decode(image)
            image_filenames.append(save_image(image_data, output_dir, i))
    else:
        print("No images found in the response")

    save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames)

def save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames):
    metrics_filename = os.path.join(output_dir, "metrics.md")
    with open(metrics_filename, "w") as metrics_file:
        metrics_file.write(f"# Generation Metrics\n")
        metrics_file.write(f"- **Prompt**: {payload['prompt']}\n")
        metrics_file.write(f"- **Steps**: {payload['steps']}\n")
        metrics_file.write(f"- **Batch Size**: {payload['batch_size']}\n")
        metrics_file.write(f"- **Inference Time**: {inference_time:.2f} seconds\n")
        metrics_file.write(f"- **Iterations per Second**: {iterations_per_second:.2f}\n")
        metrics_file.write(f"- **Generated Images**:\n")
        for filename in image_filenames:
            metrics_file.write(f"  - {filename}\n")
    print(f"Metrics saved to {metrics_filename}")

if __name__ == "__main__":
    if not check_installation():
        install_automatic1111()

    server_running = check_if_running()
    if server_running:
        generate_images("A futuristic cityscape")
    else:
        launch_automatic1111()
        generate_images("A futuristic cityscape")

