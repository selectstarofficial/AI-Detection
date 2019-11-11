import os
import os.path as osp
from pathlib import Path
import json
import abc
from glob import glob
import random
import PIL.Image
import shutil

"""
XML sample
...
<image id="0" name="MP_SEL_081701.jpg" width="1920" height="1080">
    <box label="car_plate" occluded="0" xtl="1050.56" ytl="481.52" xbr="1067.27" ybr="485.62" z_order="2">
    </box>
    <box label="car_plate" occluded="0" xtl="1370.08" ytl="496.88" xbr="1398.42" ybr="503.26" z_order="1">
    </box>
</image>
...
"""

CLASSES = {
    'car_plate': 0
}

LABELS_GLOB_PATTERN = '/mnt/hdd/Workspaces/Work/Data/number_plate/**/*.xml'
IMAGES_SEARCH_ROOT = '/mnt/hdd/Workspaces/Work/Data/number_plate_org'

OUTPUT = 'data/custom'

all_class_list = []  # for prevent duplicated warnings


class BBox:
    def __init__(self, class_name: str, x1: float, y1: float, x2: float, y2: float):
        """Each position is normalized to 0~1"""
        assert ((class_name is not None) and (class_name != "")), "class_name is empty"
        self.class_name = class_name

        if class_name in CLASSES:
            self.class_id = CLASSES[class_name]
        else:
            self.class_id = None
            if class_name not in all_class_list:
                print(f"{class_name} is not in CLASSES dict.")

        if class_name not in all_class_list:
            all_class_list.append(class_name)

        assert 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1, "Each position should be normalized to 0.0~1.0"

        self.cx = (x1+x2)/2
        self.cy = (y1+y2)/2
        self.w = abs(x2 - x1)
        self.h = abs(y2 - y1)


class Image:
    def __init__(self, image_name: str, is_train: bool, image_path: str = None, bbox_list: list = None):
        self.image_name = image_name  # Should be full name with extension
        self.image_path = image_path  # This will be later searched at once
        self.bbox_list = [] if bbox_list is None else bbox_list
        self.is_train = is_train

    def append(self, bbox: BBox):
        self.bbox_list.append(bbox)


class ImageList(list):
    def __init__(self, seq=()):
        super(ImageList, self).__init__(seq)


class Converter:
    def __init__(self, labels_glob_pattern, images_search_root, output_dir, valid_split=0.3, copy_data=True):
        self.image_list = ImageList()
        self.valid_split = valid_split

        files = self.get_label_files(labels_glob_pattern)
        print(f'{len(files)} of label files found.')

        for index, file in enumerate(files):
            print(f'{index + 1}: {file}')
            image_list = self.get_data_from_label(file, valid_split)
            self.image_list += [img for img in image_list if len(img.bbox_list) != 0]

        print(f'{len(self.image_list)} of image annotations found.')

        self.search_image_path(images_search_root)

        self.write_output(self.image_list, output_dir, copy_data)

        print("Done.")

    @staticmethod
    def get_label_files(labels_glob_pattern) -> list:
        """

        :param labels_glob_pattern: /Data/number_plate/**/*.xml
        :return: list of label files path (xml or json)
        """

        files = sorted(glob(labels_glob_pattern, recursive=True))
        return files

    def search_image_path(self, images_search_root):
        print(f'Indexing under {images_search_root}...')
        root = Path(images_search_root)
        all_paths = list(root.iterdir())

        name_path_dict = {}
        for path in all_paths:
            name_path_dict[path.name] = path

        print('Searching image file...')
        for index, image in enumerate(self.image_list):
            if image.image_path is not None:
                print(f'[{index + 1}/{len(self.image_list)}]: {image.image_name} -> image.image_path is already entered. ignoring.')
                continue

            if image.image_name in name_path_dict:
                image.image_path = name_path_dict[image.image_name]
            else:
                raise FileNotFoundError(f'{image.image_name} not found under {images_search_root}.')

            print(f'[{index + 1}/{len(self.image_list)}]: {image.image_name} -> {image.image_path}')


    # @abc.abstractstaticmethod
    # def get_data_from_label(label_file_path):
    #     """
    #
    #     :param label_file_path: json or xml path
    #     :return: 'ImageList' object
    #     """
    #     raise NotImplementedError()

    @staticmethod
    def get_data_from_label(label_file_path, valid_split):
        from xml.etree import ElementTree as ET
        element = ET.parse(label_file_path).getroot()

        image_list = ImageList()

        images = element.findall('image')
        for image in images:
            id, name, width, height = image.attrib['id'], image.attrib['name'], float(image.attrib['width']), float(
                image.attrib['height'])

            is_train = random.uniform(0, 1) > valid_split
            _image = Image(name, is_train)

            boxes = image.findall('box')
            for box in boxes:
                label, xtl, ytl, xbr, ybr = box.attrib['label'], float(box.attrib['xtl']), \
                                            float(box.attrib['ytl']), float(box.attrib['xbr']), float(box.attrib['ybr'])
                x1 = xtl / width
                y1 = ytl / height
                x2 = xbr / width
                y2 = ybr / height

                _bbox = BBox(label, x1, y1, x2, y2)
                if _bbox.class_id is None:
                    continue
                _image.append(_bbox)

            image_list.append(_image)

        return image_list

    @staticmethod
    def write_img(from_img, to_img):
        # pil_img = PIL.Image.open(from_img).convert('RGB')
        # pil_img.save(to_img)
        shutil.copy(from_img, to_img)

    @staticmethod
    def write_output(image_list, output_dir, copy_data):
        print(f'Writing data to {output_dir}')

        path = Path(output_dir)
        path.mkdir(exist_ok=True, parents=True)

        if len(os.listdir(path)) != 0:
            raise FileExistsError(f'Output directory is not empty')

        labels = path / 'labels'
        labels.mkdir(exist_ok=True)
        images = path / 'images'
        images.mkdir(exist_ok=True)

        classes = path / 'classes.names'
        train = path / 'train.txt'
        valid = path / 'valid.txt'

        classes.write_text('\n'.join(CLASSES))

        train_lines = []
        valid_lines = []

        for index, image in enumerate(image_list):
            if copy_data:
                image_path = str(images / image.image_name)
                print(f'[{index+1}/{len(image_list)}] Copy {image.image_path} -> {image_path}')
                Converter.write_img(image.image_path, image_path)

            else:
                image_path = str(image.image_path)

            label_path = (labels / (image.image_name + '.txt'))

            if image.is_train:
                train_lines.append(f'{image_path},{label_path}')
            else:
                valid_lines.append(f'{image_path},{label_path}')

            label_lines = []
            for bbox in sorted(image.bbox_list, key=lambda x: x.class_id):
                label_lines.append(f'{bbox.class_id} {bbox.cx} {bbox.cy} {bbox.w} {bbox.h}')

            label_path.write_text('\n'.join(label_lines))

        train.write_text('\n'.join(train_lines))
        valid.write_text('\n'.join(valid_lines))


if __name__ == '__main__':
    converter = Converter(LABELS_GLOB_PATTERN, IMAGES_SEARCH_ROOT, OUTPUT)
