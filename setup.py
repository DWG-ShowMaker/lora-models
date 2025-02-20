from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen-lora",
    version="0.1.0",
    author="Qwen-LoRA",
    author_email="746144374@qq.com",
    description="使用LoRA技术对Qwen 2.5模型进行微调的示例项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DWG-ShowMaker/lora-models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qwen-train=src.train:main",
            "qwen-eval=src.evaluate:main",
            "qwen-serve=src.deploy:main",
        ],
    },
) 