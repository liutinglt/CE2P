import json
import os
import cv2

json_file = os.path.join('/ssd1/liuting14/Dataset/LIP', 'annotations', 'test.json')

with open(json_file) as data_file:
    data_json = json.load(data_file)
    data_list = data_json['root']

for item in data_list:
    name = item['im_name']
    im_path = os.path.join('/ssd1/liuting14/Dataset/LIP', 'test_images', name)
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    h, w, c = im.shape
    item['img_height'] = h
    item['img_width'] = w
    item['center'] = [h/2, w/2]

with open(json_file, "w") as f:
    json.dump(data_json, f, indent=2)
