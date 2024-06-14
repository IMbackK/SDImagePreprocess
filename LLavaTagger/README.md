# LLavaTagger

LLavaTagger is a python script that tags images based on a given prompt using the [LLaVA](https://llava-vl.github.io/) multi modal llm. LLavaTagger supports using any number of gpus in ddp parralel for this task.

## How to use

first create a python venv and install the required packages into it:

	$ python -m venv venv
	$ source venv/bin/activate
	$ pip install -r requirements.txt

Then run LLavaTagger for instance like so:

	$ python LLavaTagger.py --common_description "a image of a cat, " --prompt "describe the cat in 10 to 20 words" --batch 8 --quantize --image_dir ~/cat_images

By default LLavaTagger will run in parallel on all available gpus, if this is undesriable please use the ROCR_VISIBLE_DEVICES= or CUDA_VISIBLE_DEVICES= environment variable to hide unwanted gpus

LLavaTagger will then create a meta.jsonl in the image directory sutable to be used by the scripts of [diffusers](https://github.com/huggingface/diffusers) to train stable diffusion (xl) if other formats are desired ../utils contains scripts to transform the metadata into other formats for instace for the use with [kohya](https://github.com/bmaltais/kohya_ss)

If editing the created tags is desired, [QImageTagger](https://uvos.xyz/git/uvos/QImageTagger) can be used for this purpose
