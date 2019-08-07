import argparse
from glob import glob
import json
import os
import os.path as osp
import sys
import numpy as np
import PIL.Image
import labelme

class_names = [
    'Accessories',
    'Bag',
    'Jewelry',
    'Outer',
    'Pants',
    'Shoes',
    'Skirt',
    'Tops',
    'Wholebody'
]

DATA_PATH_PREFIX = "data/"

class_name_index = {name: i for i, name in enumerate(class_names)}


def write_bbox_label(original_json, out_file):
    """
    :return: imagePath
    """
    global class_names, class_name_index

    with open(original_json) as f:
        data = json.load(f)

    img_path = data['imagePath']

    lines = []
    for shape in data['shapes']:
        if shape['shape_type'] != 'rectangle':
            print('Skipping shape: label={label}, shape_type={shape_type}'
                  .format(**shape))
            continue

        class_name = shape['label']
        (xmin, ymin), (xmax, ymax) = shape['points']

        width = int(data['imageWidth'])
        height = int(data['imageHeight'])


        assert class_name in class_name_index

        class_id = class_name_index[class_name]
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        w = abs(xmax - xmin) / width
        h = abs(ymax - ymin) / height

        if cx <= 0 or cy <= 0 or w <= 0 or h <= 0:
            raise ValueError(f"Negative Value: cx={cx}, cy={cy}, w={w}, h={h}")

        line = ' '.join((str(int(class_id)), str(cx), str(cy), str(w), str(h)))
        lines.append(line)

    if len(lines) == 0:
        raise ValueError('There is no bounding box in this image.')

    with open(out_file, 'w+') as f:
        f.write('\n'.join(lines))

    return img_path


def write_img(from_img, to_img):
    pil_img = PIL.Image.open(from_img).convert('RGB')
    pil_img.save(to_img)


def write_class_names(out_file):
    global class_names

    with open(out_file, 'w+') as f:
        f.write('\n'.join(class_names))


def main(train_dir, valid_dir, output_dir):
    os.makedirs(osp.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'images'), exist_ok=True)

    train_dir = osp.abspath(train_dir)
    valid_dir = osp.abspath(valid_dir)

    output_dataset_name = osp.split(output_dir)[-1]

    write_class_names(osp.join(output_dir, f'classes.names'))

    train_json_files = glob(osp.join(train_dir, '*.json'))
    valid_json_files = glob(osp.join(valid_dir, '*.json'))

    print(f'train: {len(train_json_files)}, valid: {len(valid_json_files)}')

    error_logs = []
    train_paths = []
    for index, json_file in enumerate(train_json_files):
        print(f'train [{index+1}/{len(train_json_files)}]: {json_file}')

        try:
            key = json_file.replace('\\', '/').split('/')[-1].replace('.json', '')
            out_label_file = osp.join(output_dir, f'labels/{key}.txt')

            img_path = write_bbox_label(json_file, out_label_file)

            write_img(osp.join(train_dir, img_path), osp.join(output_dir, f'images/{img_path.replace(".jpeg", ".jpg")}'))

            train_paths.append(osp.join(DATA_PATH_PREFIX, f'{output_dataset_name}/images/{img_path.replace(".jpeg", ".jpg")}'))

        except Exception as e:
            error_logs.append(f'train [{json_file}]: {e}')
            print(e)

    with open(osp.join(output_dir, 'train.txt'), 'w+') as f:
        f.write('\n'.join(train_paths))

    valid_paths = []
    for index, json_file in enumerate(valid_json_files):
        print(f'valid [{index+1}/{len(valid_json_files)}]: {json_file}')

        try:
            key = json_file.replace('\\', '/').split('/')[-1].replace('.json', '')
            out_label_file = osp.join(output_dir, f'labels/{key}.txt')

            img_path = write_bbox_label(json_file, out_label_file)

            write_img(osp.join(valid_dir, img_path), osp.join(output_dir, f'images/{img_path.replace(".jpeg", ".jpg")}'))

            valid_paths.append(osp.join(DATA_PATH_PREFIX, f'{output_dataset_name}/images/{img_path.replace(".jpeg", ".jpg")}'))

        except Exception as e:
            error_logs.append(f'valid [{json_file}]: {e}')
            print(e)

    with open(osp.join(output_dir, 'valid.txt'), 'w+') as f:
        f.write('\n'.join(valid_paths))

    for l in error_logs:
        print(l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train input directory path')
    parser.add_argument('--valid', help='valid input directory path')
    parser.add_argument('--output_dir', help='output dataset directory root')
    args = parser.parse_args()

    train = args.train
    valid = args.valid
    output_dir = args.output_dir

    train = train.replace('\\', '/').rstrip('/')
    valid = valid.replace('\\', '/').rstrip('/')
    output_dir = output_dir.replace('\\', '/').rstrip('/')

    assert osp.exists(train), f"{train} not exists."
    assert osp.exists(valid), f"{train} not exists."

    print('Class names:')
    print(class_names)
    print('class_names is hard coded in the source.')

    main(train, valid, output_dir)
