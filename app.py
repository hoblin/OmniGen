import gradio as gr
from PIL import Image
import os
import argparse
import random
import spaces
import json
from datetime import datetime
from OmniGen import OmniGenPipeline
from PIL import ImageDraw, ImageFont
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
    saved_path = None

    if save_images:
        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs', f'{timestamp}.png')
        img.save(output_path)
        saved_path = output_path

    return img, saved_path


def create_image_grid(grid_cells):
    # Calculate spacing and borders
    border_size = 1
    spacing = 5
    line_height = 25  # Height per line of text

    # Calculate caption height based on whether we have one or two parameters
    sample_cell = grid_cells[0][0]
    num_caption_lines = sum(1 for x in [sample_cell['x'], sample_cell['y']] if x is not None)
    caption_height = num_caption_lines * line_height + 10  # 10px padding

    # Setup font
    font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SauceCodeProNerdFontPropo-Regular.ttf')
    if not os.path.exists(font_path):
        print(f"Warning: Font not found at {font_path}, falling back to default")
        font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(font_path, 20)
        except Exception as e:
            print(f"Error loading font: {e}, falling back to default")
            font = ImageFont.load_default()

    # Calculate total dimensions including spacing and borders
    cell_width = grid_cells[0][0]['image'].size[0] + 2*border_size
    cell_height = grid_cells[0][0]['image'].size[1] + 2*border_size + caption_height

    total_width = len(grid_cells[0]) * cell_width + (len(grid_cells[0]) - 1) * spacing
    total_height = len(grid_cells) * cell_height + (len(grid_cells) - 1) * spacing

    # Create a new image with black background
    grid = Image.new('RGB', (total_width, total_height), color='black')

    # Paste images with captions
    for y, row in enumerate(grid_cells):
        for x, cell in enumerate(row):
            # Calculate position
            pos_x = x * (cell_width + spacing)
            pos_y = y * (cell_height + spacing)

            # Create white border by making a slightly larger white background
            bordered_size = (cell_width, cell_height)
            bordered_bg = Image.new('RGB', bordered_size, color='white')

            # Paste original image onto white background with border offset
            bordered_bg.paste(cell['image'], (border_size, border_size))

            # Paste bordered image onto main grid
            grid.paste(bordered_bg, (pos_x, pos_y))

            # Add text captions
            draw = ImageDraw.Draw(grid)

            # Format caption text
            caption_lines = []
            if cell['x']:
                caption_lines.append(f"X: {cell['x']['param']}: {cell['x']['value']}")
            if cell['y']:
                caption_lines.append(f"Y: {cell['y']['param']}: {cell['y']['value']}")

            # Draw each line of caption
            for i, line in enumerate(caption_lines):
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = pos_x + (cell_width - text_width) // 2
                text_y = pos_y + cell['image'].size[1] + 2*border_size + i*25  # 25 pixels between lines
                draw.text((text_x, text_y), line, fill='white', font=font)

    return grid

def apply_parameter_value(params, param_name, value):
    """Updates params dict based on parameter name and value and returns the converted value"""
    converted_value = value  # default case

    if param_name == 'inference_steps':
        converted_value = int(value)
        params['num_inference_steps'] = converted_value
    elif param_name == 'seed':
        converted_value = int(value)
        params['seed'] = converted_value
    elif param_name == 'prompt_part':
        params['prompt'] = params['prompt'].replace(..., value)
    elif param_name in ['guidance_scale', 'img_guidance_scale']:
        converted_value = float(value)
        params[param_name] = converted_value
    else:
        params[param_name] = value

    return params, converted_value

def generate_image_grid(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale,
                       inference_steps, seed, separate_cfg_infer, offload_model,
                       use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images,
                       x_param, x_values, y_param, y_values):

    if not x_values.strip() and not y_values.strip():
        return generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale,
                            inference_steps, seed, separate_cfg_infer, offload_model,
                            use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images)

    # Parse parameter values - just split CSV into arrays
    x_vals = [v.strip() for v in x_values.split(',')] if x_values.strip() else [""]
    y_vals = [v.strip() for v in y_values.split(',')] if y_values.strip() else [""]

    # Generate images for all combinations
    grid_cells = []
    for y_val in y_vals:
        row_cells = []
        for x_val in x_vals:
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

            # Apply X and Y parameters if they exist
            if x_param:
                params, x_converted = apply_parameter_value(params, x_param, x_val)
            if y_param:
                params, y_converted = apply_parameter_value(params, y_param, y_val)

            # Print generation params
            print(json.dumps({"generation_params": params}, indent=2, default=str))

            # Generate image with current parameters
            img = pipe(**params)[0]

            # Create cell with all necessary information
            cell = {
                'image': img,
                'x': {'param': x_param, 'value': x_converted} if x_param else None,
                'y': {'param': y_param, 'value': y_converted} if y_param else None
            }
            row_cells.append(cell)

            if save_images:
                os.makedirs('outputs', exist_ok=True)
                timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                output_path = os.path.join('outputs', f'{timestamp}_{x_param}_{x_val}_{y_param}_{y_val}.png')
                img.save(output_path)

        grid_cells.append(row_cells)

    # Now create_image_grid will receive complete information for each cell
    grid = create_image_grid(grid_cells)
    saved_path = None

    if save_images:
        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs', f'{timestamp}_grid.png')
        grid.save(output_path)
        saved_path = output_path

    return grid, saved_path

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
            # Parameter iteration controls
            with gr.Group():
                gr.Markdown("Parameter Iteration (optional)")
                with gr.Row():
                    x_param = gr.Dropdown(
                        choices=[("Select parameter", None),
                                ('Seed', 'seed'),
                                ('Prompt Part', 'prompt_part'),
                                ('Steps', 'inference_steps'),
                                ('Guidance Scale', 'guidance_scale'),
                                ('Image Guidance Scale', 'img_guidance_scale')],
                        label="X:",
                        value=None,
                        scale=1
                    )
                    x_values = gr.Textbox(
                        placeholder="Values (comma-separated)",
                        label="X values",
                        scale=2
                    )
                with gr.Row():
                    y_param = gr.Dropdown(
                        choices=[("Select parameter", None),
                                ('Seed', 'seed'),
                                ('Prompt Part', 'prompt_part'),
                                ('Steps', 'inference_steps'),
                                ('Guidance Scale', 'guidance_scale'),
                                ('Image Guidance Scale', 'img_guidance_scale')],
                        label="Y:",
                        value=None,
                        scale=1
                    )
                    y_values = gr.Textbox(
                        placeholder="Values (comma-separated)",
                        label="Y values",
                        scale=2
                    )

            with gr.Column():
                # output image
                save_images = gr.Checkbox(label="Save generated images", value=True)
                output_image = gr.Image(label="Output Image")
                download_link = gr.File(label="Download Generated Image")


    # click
    generate_button.click(
        generate_image_grid,
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
            x_param,
            x_values,
            y_param,
            y_values,
        ],
        outputs=[output_image, download_link],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the OmniGen')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    args = parser.parse_args()

    # launch
    demo.launch(share=args.share)
