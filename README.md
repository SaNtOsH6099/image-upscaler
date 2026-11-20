# image-upscaler

Upscales images to twice their size. It uses Real-ESRGAN x2plus, a free
model that runs on CPU. No GPU. No paid service.

## How to use it

1. Put your images in the `input/` folder.
2. Open the Actions tab on GitHub.
3. Run the "Upscale images 2x" workflow.
4. When it finishes, download the upscaled images from the artifacts
   on the run page. Each file comes back as its own artifact.

One image runs per job, and up to 20 jobs run at the same time. A
batch of 60 images takes about 10 minutes.

## How it works

The workflow lists the images in `input/` and starts one job per
image on GitHub's free runner. Each job installs Python packages,
runs `upscale.py` on its image, and uploads the result.

`upscale.py` runs the model in small tiles, so it uses little memory.
Output is JPEG, quality 95. The model file is in `models/`.

You can also run it on your own machine. First join the model parts
back together (GitHub's API refuses single files this big, so the
model ships split in `models/`):

```
cat models/realesrgan-x2plus.onnx.part-* > models/realesrgan-x2plus.onnx
pip install onnxruntime numpy pillow
python upscale.py input/test-grid.jpg output/test-grid-2x.jpg
```

## Cost

Free. This repo is public, so GitHub Actions gives unlimited minutes
on the standard runner (4 CPU, 16 GB). No card needed.

## License note

Real-ESRGAN is BSD-3 licensed. Images it upscales can be sold. Do
not swap in SUPIR: its license bans processing images meant for sale.
