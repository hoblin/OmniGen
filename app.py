import gradio as gr
from PIL import Image
import os
import argparse
import random
import spaces
from datetime import datetime
from OmniGen import OmniGenPipeline
import numpy as np

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1"
)

@spaces.GPU(duration=180)
def generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images):
    input_images = [img1, img2, img3]
    # Delete None
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0:
        input_images = None

    if randomize_seed:
        seed = random.randint(0, 10000000)

    output = pipe(
        prompt=text,
        input_images=input_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=img_guidance_scale,
        num_inference_steps=inference_steps,
        separate_cfg_infer=separate_cfg_infer,
        use_kv_cache=True,
        offload_kv_cache=True,
        offload_model=offload_model,
        use_input_image_size_as_output=use_input_image_size_as_output,
        seed=seed,
        max_input_image_size=max_input_image_size,
    )
    img = output[0]

    if save_images:
        # Save All Generated Images
        from datetime import datetime
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs', f'{timestamp}.png')
        # Save the image
        img.save(output_path)

    return img


def create_image_grid(images, values, param_name):
    # Calculate spacing and borders
    border_size = 1
    spacing = 5
    caption_height = 60  # Increased for larger font

    # Calculate total dimensions including spacing and borders
    width = sum(img.size[0] + 2*border_size for img in images) + spacing * (len(images) - 1)
    height = max(img.size[1] for img in images) + 2*border_size + caption_height

    # Create a new image with black background
    grid = Image.new('RGB', (width, height), color='black')

    # Paste images
    x_offset = 0
    for img, value in zip(images, values):
        # Create white border by making a slightly larger white background
        bordered_size = (img.size[0] + 2*border_size, img.size[1] + 2*border_size)
        bordered_bg = Image.new('RGB', bordered_size, color='white')

        # Paste original image onto white background with 1px offset
        bordered_bg.paste(img, (border_size, border_size))

        # Paste bordered image onto main grid
        grid.paste(bordered_bg, (x_offset, 0))

        # Add text caption
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        # Draw parameter value centered under each image in white
        text = f"{param_name}={value}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (bordered_size[0] - text_width) // 2
        draw.text((text_x, img.size[1] + 2*border_size + 10), text, fill='white', font=font)

        x_offset += bordered_size[0] + spacing

    return grid

def generate_image_grid(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale,
                       inference_steps, seed, separate_cfg_infer, offload_model,
                       use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images,
                       param_to_iterate, param_values):

    if not param_values.strip():
        # If no parameter values provided, just generate one image normally
        return generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale,
                            inference_steps, seed, separate_cfg_infer, offload_model,
                            use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images)

    # Parse parameter values
    try:
        if param_to_iterate == 'prompt_part':
            values = [v.strip() for v in param_values.split(',')]
        else:
            values = [float(v.strip()) for v in param_values.split(',')]
    except ValueError:
        return None, "Error: Parameter values must be comma-separated numbers (or text for prompt_part)"

    # Generate an image for each parameter value
    images = []
    for value in values:
        # Create parameters dict with the current iteration value
        params = {
            'prompt': text,
            'input_images': [img for img in [img1, img2, img3] if img is not None],
            'height': height,
            'width': width,
            'guidance_scale': guidance_scale,
            'img_guidance_scale': img_guidance_scale,
            'num_inference_steps': inference_steps,
            'seed': seed if not randomize_seed else random.randint(0, 10000000),
            'separate_cfg_infer': separate_cfg_infer,
            'offload_model': offload_model,
            'use_input_image_size_as_output': use_input_image_size_as_output,
            'max_input_image_size': max_input_image_size,
        }

        # Update the parameter being iterated
        if param_to_iterate == 'inference_steps':
            params['num_inference_steps'] = int(value)
        elif param_to_iterate == 'seed':
            params['seed'] = int(value)
        elif param_to_iterate == 'prompt_part':
            params['prompt'] = text.replace(values[0], value)
        else:
            params[param_to_iterate] = value

        # Generate image with current parameters
        img = pipe(**params)[0]
        images.append(img)

        if save_images:
            os.makedirs('outputs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            output_path = os.path.join('outputs', f'{timestamp}_{param_to_iterate}_{value}.png')
            img.save(output_path)

    grid = create_image_grid(images, values, param_to_iterate)

    if save_images:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        grid_path = os.path.join('outputs', f'{timestamp}_grid_{param_to_iterate}.png')
        grid.save(grid_path)

    return grid

# Gradio
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
    with gr.Row():
        with gr.Column():
            # text prompt
            prompt_input = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image", placeholder="Type your prompt here..."
            )

            with gr.Row(equal_height=True):
                # input images
                image_input_1 = gr.Image(label="<img><|image_1|></img>", type="filepath")
                image_input_2 = gr.Image(label="<img><|image_2|></img>", type="filepath")
                image_input_3 = gr.Image(label="<img><|image_3|></img>", type="filepath")

            # slider
            height_input = gr.Slider(
                label="Height", minimum=128, maximum=2048, value=768, step=16
            )
            width_input = gr.Slider(
                label="Width", minimum=128, maximum=2048, value=576, step=16
            )

            guidance_scale_input = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.5, step=0.1
            )

            img_guidance_scale_input = gr.Slider(
                label="img_guidance_scale", minimum=1.0, maximum=2.0, value=1.6, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=50, step=1
            )

            seed_input = gr.Slider(
                label="Seed", minimum=0, maximum=2147483647, value=42, step=1
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

            max_input_image_size = gr.Slider(
                label="max_input_image_size", minimum=128, maximum=2048, value=1024, step=16
            )

            separate_cfg_infer = gr.Checkbox(
                label="separate_cfg_infer", info="Whether to use separate inference process for different guidance. This will reduce the memory cost.", value=True,
            )
            offload_model = gr.Checkbox(
                label="offload_model", info="Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation", value=False,
            )
            use_input_image_size_as_output = gr.Checkbox(
                label="use_input_image_size_as_output", info="Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance", value=False,
            )


        with gr.Column():
            # generate
            generate_button = gr.Button("Generate Image")

            with gr.Column():
                # output image
                output_image = gr.Image(label="Output Image")
                save_images = gr.Checkbox(label="Save generated images", value=True)
            # Add parameter iteration controls
            param_to_iterate = gr.Dropdown(
                choices=['inference_steps', 'guidance_scale', 'img_guidance_scale', 'seed', 'prompt_part'],
                label="Parameter to iterate (optional)",
                value=None
            )
            param_values = gr.Textbox(
                label="Parameter values (comma-separated)",
                placeholder="e.g., 20,40,60 for steps",
                value=""
            )

    # click
    generate_button.click(
        generate_image_grid,  # Changed from generate_image to generate_image_grid
        inputs=[
            prompt_input,
            image_input_1,
            image_input_2,
            image_input_3,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            num_inference_steps,
            seed_input,
            separate_cfg_infer,
            offload_model,
            use_input_image_size_as_output,
            max_input_image_size,
            randomize_seed,
            save_images,
            param_to_iterate,  # New input
            param_values,      # New input
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the OmniGen')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    args = parser.parse_args()

    # launch
    demo.launch(share=args.share)
