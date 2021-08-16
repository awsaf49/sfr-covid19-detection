import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def submit_csv(results, submission_name):
    xyxy = np.array(results['xyxy'], dtype=np.int64)
    hw = np.array(results['hw'], dtype=np.int64)
    df_hw = np.array(results['df_hw'], dtype=np.int64)

    # scaling
    xyxy[:, :3:2] = (xyxy[:, :3:2]/hw[:,1:])*df_hw[:,1:]    #xmin xmax / width
    xyxy[:, 1::2] = (xyxy[:, 1::2]/hw[:,:1])*df_hw[:,:1]    #ymin ymax / height


    submission = {
        'image_id':results['image_id'], 'class':results['class'], 'score':results['score'],
        'xmin':xyxy[:,0], 'ymin':xyxy[:,1], 'xmax':xyxy[:,2], 'ymax':xyxy[:,3],
        'width':df_hw[:,1], 'height':df_hw[:, 0]
    }

    df = pd.DataFrame(submission)
    df.to_csv(submission_name, index=False)
    print(f'Saved {submission_name}')


def detect(opt):
    print(f'Using img-size {opt.img_size}, conf-thres {opt.conf_thres}, iou-thres {opt.iou_thres}')
        
    # saving csv submission format
    if opt.csv_format:
        detect_df = pd.read_csv(opt.detect_df)
        results = {
            'image_id': [],
            'score': [],
            'class': [],
            'xyxy':[],
            'hw':[],
            'df_hw':[]
        }

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = opt.save_img and not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device, ensemble_mode=opt.ensemble_mode)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    print('\n== Predicting ==')
    total = dataset.nf
    pbar = tqdm(total=total,leave=True, position=0,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b} [{elapsed}<{remaining}, {rate_fmt}]',
                    desc='progress ')
    total_bbox = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
            method=opt.method, img_size=imgsz, sigma=opt.sigma, merge_nms=opt.merge_nms,
            max_det=opt.max_det
        )
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # name = str(Path(path).name)
        # pbar.set_description(f'{name} '+'%gx%g ' % img.shape[2:])
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            total_bbox += len(det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if opt.csv_format:
                        image_id = Path(p).name.split('.')[0] + '_image'
                        # print(image_id)
                        w, h = detect_df.loc[detect_df.image_id==image_id,['width', 'height']].values[0]

                        conf_score = '%.4f' % (conf)
                        label_with_cls = '%s' % (names[int(cls)])
                        results['image_id'].append(image_id)
                        results['score'].append(conf_score)
                        results['class'].append(label_with_cls)
                        results['xyxy'].append(xyxy)
                        results['hw'].append((im0.shape[0], im0.shape[1]))
                        results['df_hw'].append((h, w)) 

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            pbar.update(1)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    pbar.close()
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if opt.csv_format:
        submit_csv(results, opt.submission_name)
        
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'Total BBoxes: {total_bbox:,}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    # custom
    parser.add_argument('--method', type=str, default='nms', help='nms/ snms/ wbf / nmw')
    parser.add_argument('--merge-nms', action='store_true', help='use merge nms')
    parser.add_argument('--sigma', type=float, default=0.1, help='needed for soft-nms')
    parser.add_argument('--ensemble-mode', type=str, default='nms', help='mean, max, nms')
    parser.add_argument('--csv-format', action='store_true', help='save results in csv in (xmin, ymin, xmax, ymax) format')
    parser.add_argument('--submission-name', type=str, default='submission.csv', help='name of submission csv')
    parser.add_argument('--detect_df', type=str, default='test_df.csv', help='name of test csv NEEDED for csv format')
    parser.add_argument('--save-img', type=int, default=1,help='save predicted img or not, to save time set 0')
    parser.add_argument('--weights_dirs', nargs='+', type=str, default='', help='root directory of all models')
    parser.add_argument('--use-od', action='store_true', help='use flexible yolov5 models')

    opt = parser.parse_args()
    # print(opt)
    # check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
    if len(opt.weights_dirs)!=0:
        print('\n== Searching weights ==')
        weights = []
        pbar = tqdm(opt.weights_dirs, leave=True, position=0,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b} [{elapsed}<{remaining}, {rate_fmt}]',
                    desc='progress ')
        for weight_dir in pbar:
            weights+=glob(os.path.join(weight_dir, '**/*pt'), recursive=True)
        weights = [w for w in weights if 'last.pt' not in w] # filter out last.pt
        opt.weights = weights
        print(f'Total Weights : {len(weights)}', end='\n\n')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
