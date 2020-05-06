import mmcv
import json
import argparse
import os.path as osp


def convert_annotations(path, kind):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = [
        {'supercategory': kind, 'id': 1, 'name': kind},
    ]
    lines = open(path).readlines()
    for i, line in enumerate(lines):
        print(f'{i}/{len(lines)}')
        image_info = json.loads(line)
        file_name = image_info['ID'] + '.jpg'
        image = mmcv.imread(osp.join(osp.dirname(path), 'Images', file_name))
        out_json['images'].append({
            'file_name': file_name,
            'width': image.shape[1],
            'height': image.shape[0],
            'id': img_id
        })
        for anno_info in image_info['gtboxes']:
            letter = kind[0]
            x_min, y_min, width, height = anno_info[f'{letter}box']
            # We fix small oversize of boxes for head and visible markup.
            # It is declined for full markup as mentioned in original paper.
            if kind in ('visible', 'head'):
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                width = min(width, image.shape[1] - x_min)
                height = min(height, image.shape[0] - y_min)
            out_json['annotations'].append({
                'iscrowd': anno_info['extra'].get('ignore', 0),
                'image_id': img_id,
                'bbox': [x_min, y_min, width, height],
                'area': float(anno_info[f'{letter}box'][2]) * float(anno_info[f'{letter}box'][3]),
                'segmentation': [[]],
                'category_id': 1,
                'id': ann_id
            })
            ann_id += 1
        img_id += 1
    split = path.rfind('_')
    mmcv.dump(out_json, f'{path[:split]}_{kind}{path[split:-5]}.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CrowdHuman to COCO format')
    parser.add_argument('--path', help='CrowdHuman data path')
    parser.add_argument('--kind', help='head|visible|full')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.kind in ('head', 'visible', 'full')
    convert_annotations(osp.join(args.path, 'annotation_train.odgt'), args.kind)
    convert_annotations(osp.join(args.path, 'annotation_val.odgt'), args.kind)


if __name__ == '__main__':
    main()
