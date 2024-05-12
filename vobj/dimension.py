from PIL.Image import Image


E2P = 0.5
"""
The ratio between the efficient and prompting image size.
"""

def resize_efficient(img: Image) -> Image:
    width, height = img.size
    ratio = width / height

    if ratio == 16. / 9.:
        return img.resize((320, 180))
    else:
        return img.resize((320, 240))
    

def resize_prompting(img: Image) -> Image:
    width, height = img.size
    ratio = width / height

    if ratio == 16. / 9.:
        return img.resize((640, 360))
    else:
        return img.resize((640, 480))