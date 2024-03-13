import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.paths_internal import default_output_dir
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.modules import model_management

import tempfile
import time
import random
import string

import numpy as np
import rembg
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import gc

#===========================================================================================================
#===========================================================================================================
#=== MOVE TO SYSTEM OR UTILS ===============================================================================
#===========================================================================================================

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model_root = os.path.join(models_path, 'TripoSR')
os.makedirs(model_root, exist_ok=True)
triposr_model_filenames = []

def get_rembg_model_choices():
    # List of available models. 
    return [
        "dis_anime",
        "dis_general_use",
        "silueta",
        "u2net_cloth_seg", 
        "u2net_human_seg", 
        "u2net", 
        "u2netp", 
    ]
        # "sam", #- FIXME - not currently working

def update_model_filenames():
    global triposr_model_filenames
    triposr_model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return triposr_model_filenames

triposr_model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
triposr_model.renderer.set_chunk_size(8192)
triposr_model.to(device) #- FIXME - Do not load model at start of program

def remove_model_from_memory(model, device):
    # Delete the model
    del model
    
    # If the device is a CUDA device, clear the CUDA memory cache
    if 'cuda' in device:
        torch.cuda.empty_cache()
    
    # Collect garbage to free up memory
    gc.collect()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")
    
def check_cutout_image(processed_image):
    if processed_image is None:
        raise gr.Error("No cutout image uploaded!")

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

def generate_random_filename(extension=".txt"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"{timestamp}-{random_string}{extension}"
    return filename

def write_obj_to_triposr(obj_data, filename=None):
    triposr_folder = os.path.join(default_output_dir, 'TripoSR')
    os.makedirs(triposr_folder, exist_ok=True)  # Ensure the directory exists

    if filename is None:
        filename = generate_random_filename('.obj')  # Implement or use an existing function to generate a unique filename

    full_path = os.path.join(triposr_folder, filename)

    # Assuming obj_data is a string containing the OBJ file data
    with open(full_path, 'w') as file:
        file.write(obj_data)

    return full_path

def generate(image, resolution, threshold):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    scene_codes = triposr_model(image, device=device)
    mesh = triposr_model.extract_mesh(scene_codes, resolution=int(resolution), threshold=float(threshold))[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    # Convert the mesh to a string or use a method to directly get the OBJ data
    obj_data = mesh.export(file_type='obj')  # This line might need adjustment based on how your mesh object works

    # Now save using the new function
    mesh_path = write_obj_to_triposr(obj_data)  # You could specify a filename if you want

    # Extract just the filename from the path
    filename = os.path.basename(mesh_path)

    relative_mesh_path = "output/TripoSR/" + filename

    return mesh_path, relative_mesh_path

def triposr_console_messaging(message):
    if message == "prep-start":
        print("TripoSR prepocessing has started.")
    if message == "prep-end":
        print("TripoSR prepocessing has finished.")
    if message == "rend-start":
        print("TripoSR rendering has started.")
    if message == "rend-end":
        print("TripoSR rendering has finished.")
        

#===========================================================================================================
#===========================================================================================================
#===========================================================================================================


def on_ui_tabs():
    with gr.Blocks() as model_block:
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Input Image",
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                            elem_id="content_image",
                        )
                        
                    with gr.Column():
                        processed_image = gr.Image(
                            label="Processed Image", 
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                            elem_id="cutout_image",
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")
                with gr.Row():
                    with gr.Column():
                        submit_preprocess = gr.Button("Preprocess Only", elem_id="preprocess", variant="secondary")
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

                    with gr.Column():
                        submit_postprocess = gr.Button("Render Only", elem_id="postprocess", variant="secondary")
                        gr.Markdown("### **Render Settings**\n")
                        filename = gr.Dropdown(
                            label="TripoSR Checkpoint Filename",
                            choices=triposr_model_filenames,
                            value=triposr_model_filenames[0] if len(triposr_model_filenames) > 0 else None)
                        # resolution = gr.Slider(
                        #     label="Resolution",
                        #     minimum=16,
                        #     maximum=2048,
                        #     value=256,
                        #     step=16,
                        # )
                        resolution2 = gr.Slider(
                            label="Mesh Resolution",
                            minimum=16,
                            maximum=2048,
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
                        chunking = gr.Slider( #- FIXME - Currently does nothing. Does it actually do anything at all? I don't know. It doesn't appear to affect much in tests.
                            label="Chunking",
                            minimum=128,
                            maximum=16384,
                            value=8192,
                            step=128,
                        )
                        
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("TripoSR Result"):
                        output_model = gr.Model3D(
                            label="Output Model",
                            interactive=False,
                            elem_id="triposrCanvas"
                        )

                        obj_file_path = gr.Textbox(visible=False, elem_id="obj_file_path")  # Hidden textbox to pass the OBJ file path

                    # with gr.Tab("Test"):
                    #     subject = gr.Textbox(placeholder="subject")
                    #     verb = gr.Radio(["ate", "loved", "hated"])
                    #     object = gr.Textbox(placeholder="object")
                    #     output = gr.Textbox(label="verb")
                    #     reverse_btn = gr.Button("Reverse sentence.")
                    #     reverse_btn.click(
                    #         # None, [subject, verb, object], output, _js="(s, v, o) => o + ' ' + v + ' ' + s"
                    #         None, [subject, verb, object], None, _js='''
                    #             (s, v, o) => { 
                    #                 console.log(o + ' ' + v + ' ' + s); 
                    #             }
                    #         '''
                    #     )

                    with gr.Tab("PoSR"):
                        load_obj_btn = gr.Button("Load Generated Model")
                        load_obj_btn.click(
                            None, [obj_file_path], None, _js='''
                                (objFilePath) => { 
                                    createScene(objFilePath);
                                    engine.runRenderLoop(function() {
                                        scene.render();
                                    });
                                    engine.resize(); 
                                }
                            '''
                        )

                        gr.HTML('''
                            <canvas id="babylonCanvas"></canvas>
                        ''')

                        model_block.load(
                            _js = '''
                                function babylonCanvasLoader() {                                
                                    let babylon_script = document.createElement('script');       
                                    babylon_script.src = 'https://cdn.babylonjs.com/babylon.js';
                                    babylon_script.onload = function(){
                                        let babylon_loaders_script = document.createElement('script');       
                                        babylon_loaders_script.src = 'https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js';
                                        babylon_loaders_script.onload = function(){
                                            // Access OBJFileLoader through the BABYLON namespace and enable vertex colors
                                            BABYLON.OBJFileLoader.IMPORT_VERTEX_COLORS = true;
                                           
                                            let babylonCanvasScript = document.createElement('script');
                                            babylonCanvasScript.innerHTML = `
                                                var canvas = document.getElementById('babylonCanvas');
                                                canvas.addEventListener('wheel', function(event) {
                                                    event.preventDefault();
                                                }, { passive: false });
                                           
                                                var engine = new BABYLON.Engine(canvas, true);
                                                var camera; 
                                                var scene
                                           
                                                function createScene(objFile) {
                                                    // Check if a scene already exists and dispose of it if it does
                                                    if (window.scene) {
                                                        window.scene.dispose();
                                                    }

                                                    scene = new BABYLON.Scene(engine);
                                                    scene.clearColor = new BABYLON.Color3.White();
                                           
                                                    camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 10, new BABYLON.Vector3(0, 0, 0), scene, 0.1, 10000);
                                                    camera.attachControl(canvas, true);
                                                    camera.wheelPrecision = 50;
                                           
                                                    var light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
                                                    light.intensity = 1;
                                           
                                                    // Initialize GizmoManager here
                                                    var gizmoManager = new BABYLON.GizmoManager(scene);
                                                    gizmoManager.positionGizmoEnabled = true;
                                                    gizmoManager.rotationGizmoEnabled = true;
                                                    gizmoManager.scaleRatio = 2
                                           
                                                    // Load the OBJ file
                                                    BABYLON.SceneLoader.ImportMesh("", "", "file=" + objFile, scene, function (newMeshes) {
                                                        //camera.target = newMeshes[0];
                                                        camera.target = new BABYLON.Vector3(0, 0, 0); // Keeps the camera focused on the origin
                                                      
                                                        // Define your desired scale factor
                                                        var scaleFactor = 8; // Example: Scale up by a factor of 2
                                                        // Apply a material to all loaded meshes that uses vertex colors
                                                        newMeshes.forEach(mesh => {
                                                            mesh.scaling = new BABYLON.Vector3(scaleFactor, scaleFactor, scaleFactor);
                                                        });
                                                        // Attach the first loaded mesh to the GizmoManager
                                                        if(newMeshes.length > 0) {
                                                            gizmoManager.attachToMesh(newMeshes[0]);
                                                        }
                                                    });
                                           
                                                    return scene;
                                                };
                                           
                                                window.addEventListener('resize', function() {
                                                    engine.resize(); 
                                                });
                                            `
                                            document.head.appendChild(babylonCanvasScript);
                                        };    
                                        document.head.appendChild(babylon_loaders_script);
                                    };    
                                    document.head.appendChild(babylon_script);
                                           
                                    let babylonCanvasStyle = document.createElement('style');
                                    babylonCanvasStyle.innerHTML = `
                                        #babylonCanvas {
                                            width: 100%;
                                            height: 100%;
                                            touch-action: none;
                                        }
                                    `
                                    document.head.appendChild(babylonCanvasStyle);
                                }
                            '''
                        )

                        # scene_background_image = gr.Image(
                        #     label="Scene Background",
                        #     image_mode="RGBA",
                        #     sources="upload",
                        #     type="pil",
                        #     elem_id="scene_background_image",
                        # )
                        
                        save_png_width = gr.Slider(
                            label="Image Width",
                            minimum=0,
                            maximum=2048,
                            value=512,
                            step=1,
                        )
                        save_png_height = gr.Slider(
                            label="Image Height",
                            minimum=0,
                            maximum=2048,
                            value=512,
                            step=1,
                        )
                               
                        save_png_btn = gr.Button("Save Current View to PNG")
                        save_png_btn.click(
                            None, [obj_file_path, save_png_width, save_png_height], None, _js='''
                                (objFilePath, save_png_width, save_png_height) => { 
                                    // Export to PNG button functionality
                                    BABYLON.Tools.CreateScreenshotUsingRenderTarget(engine, camera, { width: save_png_width, height: save_png_height }, function(data) {
                                        // Create a link and set the URL as the data returned from CreateScreenshot
                                        var link = document.createElement('a');
                                        link.download = 'scene.png';
                                        link.href = data;
                                        link.click();
                                    });
                                }
                            '''
                        )                

            submit_preprocess.click(
                fn=check_input_image, 
                inputs=[input_image],
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

            submit_postprocess.click(
                fn=check_cutout_image, 
                inputs=[processed_image]
            ).success(
                fn=generate,
                inputs=[processed_image, resolution2, threshold],
                outputs=[output_model, obj_file_path]
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
                inputs=[processed_image, resolution2, threshold],
                outputs=[output_model, obj_file_path]
            )

    return [(model_block, "TripoSR", "TripoSR")]


update_model_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
