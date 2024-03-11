import subprocess
import os, sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

# Current version of your extension
current_version = '1.0'

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(BASE_PATH, "requirements.txt")
models_dir = os.path.join(models_path, "TripoSR")
model_url = "https://huggingface.co/stabilityai/TripoSR/blob/main/model.ckpt"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)
installation_marker = os.path.join(BASE_PATH, ".install_complete")

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def is_installed(package: str, version: str | None = None, strict: bool = True):
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (strict and installed_version != version) or (not strict and pv.parse(installed_version) < pv.parse(version)):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

# Check if installation needs to be updated
needs_installation = True
if os.path.exists(installation_marker):
    with open(installation_marker, 'r') as f:
        installed_version = f.read().strip()
    if installed_version == current_version:
        needs_installation = False
    else:
        print(f"Updating TripoSR-webui from version {installed_version} to {current_version}.")

if needs_installation:
    # Your installation or update logic here
    # Ensure the directory for models exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        download(model_url, model_path)

    with open(req_file) as file:
        for package in file:
            package = package.strip()
            package_version = None
            strict = True
            if "==" in package:
                package_version = package.split('==')[1]
            elif ">=" in package:
                package_version = package.split('>=')[1]
                strict = False
            if not is_installed(package, package_version, strict):
                pip_install(package)

    # After successful installation/update, write the current version to the marker file
    with open(installation_marker, 'w') as f:
        f.write(current_version)
else:
    print("Launching with TripoSR-webui installed...")
