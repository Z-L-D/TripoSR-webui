import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config

# import logging
import tempfile
# import time

import numpy as np
import rembg
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


model_root = os.path.join(models_path, 'TripoSR')
os.makedirs(model_root, exist_ok=True)
triposr_model_filenames = []

def get_rembg_model_choices():
    # List of available models. This could be dynamic based on downloaded models
    return [
        "dis_anime",
        "dis_general_use",
        "sam", #!- FIXME !!!!!!!!!! not currently working
        "silueta",
        "u2net_cloth_seg", 
        "u2net_human_seg", 
        "u2net", 
        "u2netp", 
    ]

def update_model_filenames():
    global triposr_model_filenames
    triposr_model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return triposr_model_filenames

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

# rembg_session = rembg.new_session(model_name="dis_general_use")

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(
    input_image, 
    rembg_model,
    do_remove_background, 
    foreground_ratio,
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=0
):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(
            image,
            rembg.new_session(model_name=rembg_model),
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size
        )
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, resolution, threshold):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=int(resolution), threshold=float(threshold))[0]
    mesh = to_gradio_3d_orientation(mesh)
    mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(mesh_path.name)
    return mesh_path.name

def dummy_function():
    # This function won't do anything but is needed to bind the button click event
    pass

def on_ui_tabs():
    with gr.Blocks() as model_block:
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
                        gr.Markdown("### **Preprocess Settings**\n")
                        rembg_model_dropdown = gr.Dropdown(
                            label="Cutout Model",
                            choices=get_rembg_model_choices(),
                            value="dis_general_use",  # Default value
                        )
                        do_remove_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                        foreground_ratio = gr.Slider(
                            label="Subject Zoom",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                        )
                        alpha_matting = gr.Checkbox(
                            label="Enable Alpha Matting", value=False
                        )
                        gr.Markdown("*Improves edge and transparency handling*")
                        alpha_matting_foreground_threshold = gr.Slider(
                            label="Alpha Matting Foreground Threshold",
                            minimum=0,
                            maximum=255,
                            value=240,
                            step=1,
                        )
                        alpha_matting_background_threshold = gr.Slider(
                            label="Alpha Matting Background Threshold",
                            minimum=0,
                            maximum=255,
                            value=10,
                            step=1,
                        )
                        alpha_matting_erode_size = gr.Slider(
                            label="Alpha Matting Erode Size",
                            minimum=0,
                            maximum=50,
                            value=0,
                            step=1,
                        )
                with gr.Row():
                    submit_preprocess = gr.Button("Preprocess Only", elem_id="preprocess", variant="secondary")
                gr.Markdown("\n")
                with gr.Row():
                    with gr.Group():
                        gr.Markdown("### **Render Settings**\n")
                        filename = gr.Dropdown(
                            label="TripoSR Checkpoint Filename",
                            choices=triposr_model_filenames,
                            value=triposr_model_filenames[0] if len(triposr_model_filenames) > 0 else None)
                        refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                        refresh_button.click(
                            fn=lambda: gr.update(choices=update_triposr_model_filenames),
                            inputs=[], outputs=filename
                        )
                        resolution = gr.Slider(
                            label="Resolution",
                            minimum=16,
                            maximum=512,
                            value=256,
                            step=16,
                        )
                        threshold = gr.Slider(
                            label="Threshold",
                            minimum=0,
                            maximum=100,
                            value=25,
                            step=0.1,
                        )
                        chunking = gr.Slider(
                            label="Chunking",
                            minimum=128,
                            maximum=16384,
                            value=8192,
                            step=128,
                        )

                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Column():
                with gr.Row():
                    output_model = gr.Model3D(
                        label="Output Model",
                        interactive=False,
                        elem_id="triposr_canvas"
                    )
                gr.HTML('''
                    <button id="my_btn">Hello</button>
                    <div id="result"></div>
                    <canvas id="babylonCanvas"></canvas>
                ''')
                model_block.load(_js = '''
                    function test() {
                        let jq_script = document.createElement('script');
                        jq_script.src = 'https://code.jquery.com/jquery-3.6.0.min.js';
                        jq_script.onload = function() {
                            console.log('jQuery loaded successfully!');
                            // Add your jQuery code here
                            $(function() {
                                $("#my_btn").click(function() {
                                    $("#result").text("Button clicked!");
                                });
                            });
                        };
                        document.head.appendChild(jq_script);
                                 
                        let babylon_script = document.createElement('script');       
                        babylon_script.src = 'https://cdn.babylonjs.com/babylon.js';
                        babylon_script.onload = function(){};    
                        document.head.appendChild(babylon_script);
                                 
                        let babylon_loaders_script = document.createElement('script');       
                        babylon_loaders_script.src = 'https://preview.babylonjs.com/loaders/babylonjs.loaders.min.js';
                        babylon_loaders_script.onload = function(){};    
                        document.head.appendChild(babylon_loaders_script);
                                 
                        let babylonCanvasStyle = document.createElement('style');
                        babylonCanvasStyle.innerHTML = `
                            #babylonCanvas {
                                width: 100%;
                                height: 100%;
                                touch-action: none;
                            }
                        `
                        document.head.appendChild(babylonCanvasStyle);
                                 
                        let babylonCanvasScript = document.createElement('script');
                        babylonCanvasScript.innerHTML = `
                            window.addEventListener('DOMContentLoaded', function() {
                                var canvas = document.getElementById('renderCanvas');
                                var engine = new BABYLON.Engine(canvas, true);

                                var createScene = function() {
                                    var scene = new BABYLON.Scene(engine);
                                    scene.clearColor = new BABYLON.Color3.White();

                                    var camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 10, new BABYLON.Vector3(0, 0, 0), scene);
                                    camera.attachControl(canvas, true);

                                    var light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
                                    light.intensity = 0.7;

                                    // Load the OBJ file
                                    BABYLON.SceneLoader.ImportMesh("", "", "your-obj-file-path.obj", scene, function (newMeshes) {
                                        camera.target = newMeshes[0];
                                    });

                                    return scene;
                                };

                                var scene = createScene();

                                engine.runRenderLoop(function() {
                                    scene.render();
                                });

                                window.addEventListener('resize', function() {
                                    engine.resize();
                                });
                            });
                        `
                        document.head.appendChild(babylonCanvasScript);
                    }
                ''')
                #! FIXME -- Uncaught TypeError: d.FlowGraphSceneReadyEventBlock is undefined

            submit_preprocess.click(
                fn=check_input_image, inputs=[input_image]
            ).success(
                fn=preprocess,
                inputs=[
                    input_image, 
                    rembg_model_dropdown,
                    do_remove_background, 
                    foreground_ratio,
                    alpha_matting,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size
                ],
                outputs=[processed_image]
            )
            
            submit.click(
                fn=check_input_image, inputs=[input_image]
            ).success(
                fn=preprocess,
                inputs=[
                    input_image, 
                    rembg_model_dropdown,
                    do_remove_background, 
                    foreground_ratio,
                    alpha_matting,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size
                ],
                outputs=[processed_image]
            ).success(
                fn=generate,
                inputs=[processed_image, resolution, threshold],
                outputs=[output_model]
            )

    return [(model_block, "TripoSR", "TripoSR")]


update_model_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
