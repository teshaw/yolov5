import os
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from uuid import uuid4

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages#,IMG_FORMATS,VID_FORMATS, LoadScreenshots, LoadStreams
from utils.general import (LOGGER,set_logging,
                           Profile,check_img_size,
                           non_max_suppression,
                           scale_boxes,
                           xyxy2xywh)
# from utils.general import (LOGGER,set_logging, Profile, check_file, check_imshow, check_requirements, colorstr, cv2 ,
#                            increment_path, , print_args, , strip_optimizer, )
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device#, smart_inference_mode
from utils.augmentations import letterbox

def sorter(fp,default=1E9,reverse=False):
    _,fn = os.path.split(fp)
    if "_" in fn:
        try:
            result = int(fn.split("_",maxsplit=1)[0])
        except:
            result =  default
    else:
        result = default
    if reverse:
        return default-result
    else:
        return result

class ImageDetector():
    def __init__(self,modelpath,*args,**kwargs):
        self.ROOT = Path().absolute()
        self.weights=Path(modelpath).absolute()  # model path or triton URL
        # source=self.ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
        self.data=self.ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # self.view_img=False,  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=True  # save confidences in --save-txt labels
        # self.save_crop=False  # save cropped prediction boxes
        # self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=self.ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.skip_existing=False  # Skip if image output exists
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.vid_stride=1  # video frame-rate stride
        self.image_sort=None # function to parse an image name for a sorting key.
        self.verbose=True

        self.__load_model__()

    def __load_model__(self):
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights,
                                        device=device,
                                        dnn=self.dnn,
                                        data=self.data,
                                        fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.__check_img_size__()

    def __check_img_size__(self,imgsz=None):
        if imgsz is None:
            imgsz = self.imgsz
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def __imload__(self,source):
        im = LoadImages(source,
                   img_size=self.imgsz,
                   stride=self.stride,
                   auto=self.pt,
                   image_sort=None)
        for i in im:
            yield i

    def imdetect(self,imgpath,path=None):
        '''run image detection on in memory image'''
        if type(imgpath) == type(np.zeros((0,0))):
            def dummy(im0,imgsz,stride,auto,path):
                im = letterbox(im0, imgsz, stride=stride, auto=auto)[0]  # padded resize
                if im.ndim > 2:
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                while im.ndim < 4:
                    im = np.expand_dims(im,0)
                im = np.ascontiguousarray(im)  # contiguous
                s = f"image 1/1 numpy array with declared path {path}"
                return [(path, im, im0, 0, s)]
            dataset = dummy(imgpath,self.imgsz,self.stride,self.pt,path)
        else:
            dataset = self.__imload__(imgpath)
        set_logging(verbose=self.verbose)
        seen=0
        for path, im, im0s, vid_cap, s in dataset:
            dt = [Profile(),]*3
            seen+=1
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred,
                                        self.conf_thres,
                                        self.iou_thres,
                                        self.classes,
                                        self.agnostic_nms,
                                        max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                result = DetectionResult(im0,p,det,self.names)
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
                yield result
        # Print results
        set_logging(verbose=True)
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        set_logging(verbose=self.verbose)


class DetectionResult():
    '''image and detection results'''
    def __init__(self,im0,path,detections,names,sql_params={}):
        self.img = im0
        self.path = path
        self.names = names
        self.sql_params=sql_params
        self.__detections__=detections #keep raw detections tensor.
        self.detections=[]
        self.__parse_detections__(detections)

    def __parse_detections__(self,detections):
        for *xyxy,conf,cls in reversed(detections):
            cls_name = self.names[int(cls)]
            intxyxy = [int(n//1) for n in xyxy]
            self.detections.append({'class':cls_name,
                               'confidence': float(conf),
                               'guid':str(uuid4()),
                               'xyxy':intxyxy,
                               'coordstring':str(intxyxy)})
            self.detections[-1].update(self.sql_params)
    def __str__(self):
        summary = defaultdict(lambda:0)
        for d in self.detections:
            summary[d['class']]+=1
        results = ",".join([f"{n} {c}" for c,n in summary.items()])
        return f"DetectionResult({self.path.name}: {results})"

    def to_database(self,conn):
        sql_query = """SELECT fileID FROM images"""
        sql_insert = """INSERT OR IGNORE INTO annotations
                        (fileID,type,tag,uuid,confidence,coords)
                        VALUES (:fileID,'rectangle',
                                :class,
                                :guid,
                                :confidence,
                                :coordstring);"""
        if self.detections:
            with conn as Q:
                Q.executemany(sql,self.detections)

    def write(self):
        return
        # Write results
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)



if __name__ == "__main__":
    # Test of working functionality
    # weights = r".\exclude\best.pt"
    # im_path = r".\exclude\image.png"
    # IM = ImageDetector(weights)
    # from detect import run
    # weights=r"G:\My Drive\ML\YOLO\runs\amtrak\v5x640\weights\best.pt"
    arc_weights = r"G:\My Drive\ML\YOLO\runs\train\arc\v5x640-combi2\weights\best.pt"
    # sleepersB = r"G:\My Drive\ML\RTIO\yolov5\runs\train\KIWI\sleepersB_320s\weights\best.pt"
    # sleepersC = r"G:\My Drive\ML\RTIO\yolov5\runs\train\KIWI\sleepersC_320s\weights\best.pt"
    path = r"K:\Rail\Projects\ZR0710-22 QR\Data\LSC\Run_161-20230402@060706\LSC1\normalised\0_Camera1.jpg"
    IM = ImageDetector(arc_weights)
    import cv2
    im = cv2.imread(path)
    for i in IM.imdetect(im,path="./placeholder"):
        print(i)
    # run(weights=arc_weights,
    #     source=os.path.join(path,"*.jpg"),
    #     nosave=False,
    #     agnostic_nms=True,
    #     iou_thres=0.3,
    #     conf_thres=0.2,
    #     save_txt=True,
    #     save_conf=True,
    #     save_crop=True,
    #     project=os.path.join(os.path.split(path)[0],"YOLO"),
    #     imgsz=(640,640),
    #     name=f"arcSleepers",
    #     exist_ok=True,
    #     skip_existing=True,
    #     image_sort=sorter)