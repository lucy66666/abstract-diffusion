## The code is mostly take from https://huggingface.co/blog/stable_diffusion
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse



## Put images together
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='path to pre-trained stable diffusion model')
    parser.add_argument('--input_path', type=str, default='prompts.txt',
                        help='input path to the prompts')
    parser.add_argument('--seed', type=int, default=185848191, 
                        help='random seed for generation')
    parser.add_argument('--num_images', type=int, default=9, 
                        help='number of images generated per prompt')
    parser.add_argument('--num_rows', type=int, default=3, 
                        help='number of images displayed per row') 
    parser.add_argument('--num_cols', type=int, default=3, 
                        help='number of images displayed per column')                   
    parser.add_argument('--cuda', action='store_false',
                        help='whether to use GPU')
    parser.add_argument('--save_individual', action='store_true',
                        help='whether to save individual images')
    parser.add_argument('--output_path', type=str, default='images',
                        help='output path to the images')

    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, use_auth_token=True)
    generator = torch.Generator("cuda").manual_seed(args.seed)
    if args.cuda:
        pipe = pipe.to("cuda")

    with open(args.input_path) as f:
        prompts = f.read().splitlines()

    for p in prompts:
        print("Input prompt: " + p)
        prompt = [p] * args.num_images
        images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
        grid = image_grid(images, rows=args.num_rows, cols=args.num_cols)
        grid.save(args.output_path + '/' + p + '_' + str(args.seed) + '.jpg')

        if args.save_individual:
            for idx, image in enumerate(images):
                image.save(args.output_path + '/' + p + '_' + str(args.seed) + '_' + str(idx) + '.jpg') 

if __name__ == '__main__':
    main()

