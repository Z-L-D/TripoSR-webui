import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

def update_model_filenames():
    global model_filenames
    model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return model_filenames

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh = to_gradio_3d_orientation(mesh)
    mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(mesh_path.name)
    return mesh_path.name

# @torch.inference_mode()
# @torch.no_grad()
# def predict(filename, width, height, batch_size, elevation, azimuth,
#             sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler, sampling_denoise, input_image):
#     filename = os.path.join(model_root, filename)
#     model, _, vae, clip_vision = \
#         load_checkpoint_guess_config(filename, output_vae=True, output_clip=False, output_clipvision=True)
#     init_image = numpy_to_pytorch(input_image)
#     positive, negative, latent_image = opStableZero123_Conditioning.encode(
#         clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth)
#     output_latent = opKSampler.sample(model, sampling_seed, sampling_steps, sampling_cfg,
#                                       sampling_sampler_name, sampling_scheduler, positive,
#                                       negative, latent_image, sampling_denoise)[0]
#     output_pixels = opVAEDecode.decode(vae, output_latent)[0]
#     outputs = pytorch_to_numpy(output_pixels)
#     return outputs

def on_ui_tabs():
    with gr.Blocks() as model_block:
        with ResizeHandleRow():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="TripoSR Checkpoint Filename",
                                           choices=model_filenames,
                                           value=model_filenames[0] if len(model_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_model_filenames),
                        inputs=[], outputs=filename)

        with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Tab("Model"):
                output_model = gr.Model3D(
                    label="Output Model",
                    interactive=False,
                )

    return [(model_block, "TripoSR", "TripoSR")]


update_model_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
