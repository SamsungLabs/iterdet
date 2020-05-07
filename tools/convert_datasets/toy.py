import os
import mmcv
import argparse
import numpy as np


def convert_annotations(path):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = [
        {'supercategory': '1', 'id': 1, 'name': '1'},
    ]
    file_names = tuple(filter(lambda x: 'im' in x, os.listdir(path)))
    for i, file_name in enumerate(file_names):
        print(f'{i}/{len(file_names)}')
        mask = mmcv.imread(os.path.join(path, file_name), flag='unchanged')
        out_json['images'].append({
            'file_name': file_name.replace('im', 'rgb'),
            'width': 128,
            'height': 128,
            'id': img_id
        })
        for j in range(1, np.max(mask) + 1):
            indexes = np.argwhere(mask == j)
            x_min = int(np.min(indexes[:, 1]))
            y_min = int(np.min(indexes[:, 0]))
            x_max = int(np.max(indexes[:, 1]))
            y_max = int(np.max(indexes[:, 0]))
            out_json['annotations'].append({
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'area': (x_max - x_min) * (y_max - y_min),
                'segmentation': [[]],
                'category_id': 1,
                'id': ann_id
            })
            ann_id += 1
        img_id += 1
    mmcv.dump(out_json, path + '.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Adaptis Toy V2 to COCO format')
    parser.add_argument('--path', help='Adaptis Toy V2 / V1 data path', required=True)
    parser.add_argument('--version', type=int, help='1 or 2', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.version in (1, 2)
    if args.version == 2:
        convert_annotations(os.path.join(args.path, 'train'))
        convert_annotations(os.path.join(args.path, 'val'))
        convert_annotations(os.path.join(args.path, 'test'))
    else:
        convert_annotations(os.path.join(args.path, 'augmented', 'train'))
        convert_annotations(os.path.join(args.path, 'augmented', 'test'))


if __name__ == '__main__':
    main()
