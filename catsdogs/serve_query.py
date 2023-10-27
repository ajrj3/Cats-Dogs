import requests

from catsdogs.config import CATS_DIR
from catsdogs.data import img_to_array, load_img, preprocess_img

test_file_path = CATS_DIR + '/1184.jpg'
test_img = load_img(test_file_path, target_size=(200,200))
test_arr = img_to_array(test_img)
test_arr = preprocess_img(test_arr)

resp = requests.post('http://127.0.0.1:8265/', json={'array': test_arr.tolist()})

print(resp.json())