import os
import cv2
import mmcv
import numpy as np
import pandas as pd
import brambox
from collections import defaultdict

# from .coco import CocoDataset
from .custom import CustomDataset
from .registry import DATASETS
from mmdet.utils import print_log


# @DATASETS.register_module
# class CrowdHumanDataset(CocoDataset):
#     CLASSES = ('smth',)


@DATASETS.register_module
class CrowdHumanDataset(CustomDataset):
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        results = defaultdict(lambda: {'ann': defaultdict(list)})
        image_data = {v['id']: v for v in data['images']}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            results[image_id]['filename'] = image_data[image_id]['file_name']
            results[image_id]['width'] = image_data[image_id]['width']
            results[image_id]['height'] = image_data[image_id]['height']
            bbox = annotation['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            results[image_id]['ann']['bboxes_ignore' if annotation['iscrowd'] else 'bboxes'].append(bbox)
            results[image_id]['ann']['labels_ignore' if annotation['iscrowd'] else 'labels'].append(1)
        results = list(results.values())
        for annotation in results:
            annotation['ann']['bboxes'] = np.array(annotation['ann']['bboxes'], dtype=np.float32)
            if not len(annotation['ann']['bboxes']):
                annotation['ann']['bboxes'] = np.zeros((0, 4), dtype=np.float32)
            annotation['ann']['labels'] = np.array(annotation['ann']['labels'], dtype=np.int64)
            annotation['ann']['bboxes_ignore'] = np.array(annotation['ann']['bboxes_ignore'], dtype=np.float32)
            if not len(annotation['ann']['bboxes_ignore']):
                annotation['ann']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            annotation['ann']['labels_ignore'] = np.array(annotation['ann']['labels_ignore'], dtype=np.int64)
        return results


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 iou_thr=0.5):
        # annotations to brambox
        true_df = defaultdict(list)
        for img_info in self.img_infos:
            bboxes = np.concatenate((img_info['ann']['bboxes'], img_info['ann']['bboxes_ignore']), axis=0)
            labels = np.concatenate((img_info['ann']['labels'], img_info['ann']['labels_ignore']), axis=0)
            ignores = [False] * len(img_info['ann']['bboxes']) + [True] * len(img_info['ann']['bboxes_ignore'])
            for bbox, label, ignore in zip(bboxes, labels, ignores):
                true_df['image'].append(img_info['filename'])
                true_df['class_label'].append(str(label))
                true_df['id'].append(0)
                true_df['x_top_left'].append(bbox[0])
                true_df['y_top_left'].append(bbox[1])
                true_df['width'].append(bbox[2] - bbox[0])
                true_df['height'].append(bbox[3] - bbox[1])
                true_df['ignore'].append(ignore)
        true_df = pd.DataFrame(true_df)
        true_df['image'] = true_df['image'].astype('category')

        # results to brambox
        predicted_df = defaultdict(list)
        for i, image_results in enumerate(results):
            for j, class_detection in enumerate(image_results):
                for detection in class_detection:
                    predicted_df['image'].append(self.img_infos[i]['filename'])
                    predicted_df['class_label'].append(str(j + 1))
                    predicted_df['id'].append(0)
                    predicted_df['x_top_left'].append(detection[0])
                    predicted_df['y_top_left'].append(detection[1])
                    predicted_df['width'].append(detection[2] - detection[0])
                    predicted_df['height'].append(detection[3] - detection[1])
                    predicted_df['confidence'].append(detection[4])
        predicted_df = pd.DataFrame(predicted_df)
        predicted_df['image'] = predicted_df['image'].astype('category')

        pr = brambox.stat.pr(predicted_df, true_df, iou_thr)
        ap = brambox.stat.ap(pr)
        mr_fppi = brambox.stat.mr_fppi(predicted_df, true_df, iou_thr)
        lamr = brambox.stat.lamr(mr_fppi)
        eval_results = {
            'gts': len(true_df[~true_df['ignore']]),
            'dets': len(predicted_df),
            'recall': pr['recall'].values[-1],
            'mAP': ap,
            'mMR': lamr
        }
        print_log(str(eval_results), logger)
        return eval_results

    # All functions below were used for paper visualizations.
    # To use them again pass iteration number as a label in Detector, and then to predicted_df['iteration'].
    # After this the can be called from self.evaluate.
    def plot(self, true_df, predicted_df, iou_thr):
        # update path here !
        image = mmcv.imread(os.path.join('data', 'adaptis_toy_v2', 'test', self.img_infos[0]['filename']))
        predicted_df.sort_values('confidence', ascending=False, inplace=True)
        predicted_df.reset_index(inplace=True)

        mmcv.imwrite(
            self.draw_boxes(image, true_df[~true_df['ignore']], (25, 25, 255 - 25), 25, 2),
            './work_dirs/true.png'
        )
        n_iterations = len(predicted_df['iteration'].value_counts())
        print(predicted_df['iteration'].value_counts())
        for iteration in range(n_iterations):
            predicted_df['vis'] = 'fp'
            true_df['used'] = False
            ious = brambox.stat.coordinates.iou(predicted_df, true_df)
            ioas = brambox.stat.coordinates.ioa(predicted_df, true_df)
            for i in range(len(predicted_df)):
                if predicted_df['iteration'][i] > iteration:
                    continue
                best_iou = -1
                best_true = -1
                for j in range(len(true_df)):
                    if true_df['ignore'][j]:
                        if ioas[i, j] > iou_thr:
                            predicted_df.at[i, 'vis'] = 'ignore'
                    else:
                        if not true_df['used'][j] and ious[i, j] > iou_thr and ious[i, j] > best_iou:
                            best_iou = ious[i, j]
                            best_true = j
                if best_iou > 0:
                    predicted_df.at[i, 'vis'] = 'tp'
                    true_df.at[best_true, 'used'] = True
            print(predicted_df['vis'][predicted_df['iteration'] <= iteration].value_counts())
            it_image = self.draw_boxes(image, predicted_df[
                (predicted_df['iteration'] < iteration) &
                (predicted_df['vis'] == 'tp')
            ], (25, 145, 25), 25, 2)
            color, width = ((105, 255 - 25, 255 - 25), 3) if iteration !=0 else ((25, 145, 25), 2)
            it_image = self.draw_boxes(it_image, predicted_df[
                (predicted_df['iteration'] == iteration) &
                (predicted_df['vis'] == 'tp')
                ], color, 25, width)
            mmcv.imwrite(it_image, f'./work_dirs/predicted_{iteration}.png')


    def draw_boxes(self, image, df, color, color_thr, width):
        image = np.copy(image)
        for row in df.itertuples():
            cv2.rectangle(
                image,
                (max(int(row.x_top_left), 0), max(int(row.y_top_left), 0)),
                (
                    min(int(row.x_top_left + row.width), image.shape[1] - 1),
                    min(int(row.y_top_left + row.height), image.shape[0] - 1)
                ),
                self.get_random_color(color, color_thr),
                width,
                cv2.LINE_8
            )
        return image

    def get_random_color(self, color, thr):
        return np.clip(np.random.randint(-thr, thr, 3) + color, 0, 255).tolist()

    def print_statistics(self, results):
        file_names = pd.unique(results['image'])
        print('objects/image', len(results) / len(file_names))
        counts = {0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0}
        for file_name in file_names:
            df = results[results['image'] == file_name]
            ious = brambox.stat.coordinates.iou(df, df)
            for thr in counts.keys():
                counts[thr] += (np.sum(ious > thr) - len(df)) / 2
        for thr in counts.keys():
            counts[thr] /= len(file_names)
        print(counts)
