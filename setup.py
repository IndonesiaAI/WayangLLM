import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup


stale_egg_info = Path(__file__).parent / "alignment.egg-info"
if stale_egg_info.exists():
    shutil.rmtree(stale_egg_info)


_deps = [
    "accelerate==0.23.0",
    "bitsandbytes==0.41.2.post2",
    "black==23.1.0",
    "datasets==2.14.6",
    "deepspeed==0.12.2",
    "einops>=0.6.1",
    "evaluate==0.4.0",
    "flake8>=6.0.0",
    "hf-doc-builder>=0.4.0",
    "huggingface-hub>=0.14.1,<1.0",
    "isort>=5.12.0",
    "ninja>=1.11.1",
    "numpy>=1.24.2",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft==0.6.1",
    "protobuf<=3.20.2",  # Needed to avoid conflicts with `transformers`
    "pytest",
    "safetensors>=0.3.3",
    "scipy",
    "tensorboard",
    "torch==2.1.0",
    "transformers==4.35.0",
    "trl==0.7.4",
    "jinja2>=3.0.0",
    "tqdm>=4.64.1",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["tests"] = deps_list("pytest", "parameterized")
extras["torch"] = deps_list("torch")
extras["quality"] = deps_list("black", "isort", "flake8")
extras["docs"] = deps_list("hf-doc-builder")
extras["dev"] = extras["docs"] + extras["quality"] + extras["tests"]

# core dependencies shared across the whole project - keep this to a bare minimum :)
install_requires = [
    deps["accelerate"],
    deps["bitsandbytes"],
    deps["einops"],
    deps["evaluate"],
    deps["datasets"],
    deps["deepspeed"],
    deps["huggingface-hub"],
    deps["jinja2"],
    deps["ninja"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["peft"],
    deps["protobuf"],
    deps["safetensors"],
    deps["scipy"],
    deps["tensorboard"],
    deps["tqdm"],  # progress bars in model download and training scripts
    deps["transformers"],
    deps["trl"],
]

setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.10.9",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
