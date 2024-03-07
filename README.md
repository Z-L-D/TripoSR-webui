# TripoSR-webui
A TripoSR implementation for WebUI

This is still a few revisions away from being releasable but it works in its current state. For anyone that wants to try this out, just git clone or download it into your extensions directory of webui. You will also need to remove the .no extension from install.py.no and requirements.txt.no. Following this, it should download the necessary components to start webui. Models will download when they are run for the first time. 

Current potential issues:
* I have only tested this on webui-forge at this time.
* Placing this in your extensions directory at this time will attempt to install it every time you load webui which can hang for a bit.
* The sam masking filter is not working at this time

Future updates coming:
* A button to send txt2img or img2img generations into TripoSR
* A button to save the current 3D view
* A button to send the current 3D view to img2img
* Provide the generated cutout mask to eventually perform a background recovery in img2img inpainting