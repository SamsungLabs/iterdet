import mmcv
import json
import argparse
import os.path as osp


def convert_annotations(path):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = [
        {'supercategory': 'pedestrian', 'id': 1, 'name': 'pedestrian'},
    ]
    lines = open(path).readlines()
    for i, line in enumerate(lines):
        print(f'{i}/{len(lines)}')
        file_name = f'{line.strip()}.jpg'
        annotations = open(osp.join(osp.dirname(path), 'Annotations', f'{file_name}.txt')).readlines()[1:]
        # We skip training images with 0 pedestrians.
        if osp.basename(path).startswith('train') and \
            len(tuple(filter(lambda x: int(x.split()[0]) == 1, annotations))) == 0:
            print(f'No pedestrians for {file_name}')
            continue
        image = mmcv.imread(osp.join(osp.dirname(path), 'Images', file_name))
        out_json['images'].append({
            'file_name': file_name,
            'width': image.shape[1],
            'height': image.shape[0],
            'id': img_id
        })
        for annotation in annotations:
            label, x_min, y_min, x_max, y_max = tuple(map(int, annotation.split()))
            out_json['annotations'].append({
                'iscrowd': label != 1,
                'image_id': img_id,
                'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                'area': (x_max - x_min + 1) * (y_max - y_min + 1),
                'segmentation': [[]],
                'category_id': 1,
                'id': ann_id
            })
            ann_id += 1
        img_id += 1
    mmcv.dump(out_json, f'{path[:-4]}.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert WiderPerson to COCO format')
    parser.add_argument('--path', help='WiderPerson data path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert_annotations(osp.join(args.path, 'train.txt'))
    convert_annotations(osp.join(args.path, 'val.txt'))


if __name__ == '__main__':
    main()
