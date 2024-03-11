import subprocess
import os, sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

req_file = os.path.join(BASE_PATH, "requirements.txt")

models_dir = os.path.join(models_path, "TripoSR")
model_url = "https://huggingface.co/stabilityai/TripoSR/blob/main/model.ckpt"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

# Path for the installation marker file
installation_marker = os.path.join(BASE_PATH, ".install_complete")

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

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

# Check if the installation has already been completed
if os.path.exists(installation_marker):
    sys.exit(0)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

with open(req_file) as file:
    install_count = 0
    strict = True
    for package in file:
        package_version = None
        try:
            package = package.strip()
            if "==" in package:
                package_version = package.split('==')[1]
            elif ">=" in package:
                package_version = package.split('>=')[1]
                strict = False
            if not is_installed(package, package_version, strict):
                install_count += 1
                pip_install(package)
        except Exception as e:
            print(e)
            print(f"\nERROR: Failed to install {package} - TripoSR won't start")
            raise e

# After successful installation, create a marker file
with open(installation_marker, 'w') as f:
    f.write('Installation completed.')
