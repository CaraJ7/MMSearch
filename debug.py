import datasets
import pdb
from io import BytesIO
from PIL import Image

def pil_image_to_bytes(pil_image, format='PNG'):
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

dataset = datasets.load_dataset('CaraJ/MMSearch', name='rerank', split='rerank')
image_bytes = pil_image_to_bytes(dataset[0]['query_image']) # .convert('RGB')
image_data = BytesIO(image_bytes)
image = Image.open(image_data)
pdb.set_trace()
image.save('debug.png')
print(dataset)