"""Enquadra por padding, sem esticar/cortar a referência para dimensões do modelo."""
import math
from PIL import Image


def prepare_reference(image, max_side=1024, margin=.08):
    width,height=image.size
    scale=max_side/(max(width,height)*(1+margin*2))
    inner=(max(1,round(width*scale)),max(1,round(height*scale)))
    canvas_size=tuple(max(64,math.ceil(value*(1+margin*2)/16)*16) for value in inner)
    canvas=Image.new('RGB',canvas_size,'white')
    left,top=(canvas_size[0]-inner[0])//2,(canvas_size[1]-inner[1])//2
    canvas.paste(image.resize(inner,Image.Resampling.LANCZOS),(left,top))
    return canvas,(left,top,left+inner[0],top+inner[1])


def restore_frame(result, box, original_size):
    return result.crop(box).resize(original_size,Image.Resampling.LANCZOS)
