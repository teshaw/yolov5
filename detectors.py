import os
from detect import run

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

if __name__ == "__main__":
    # Test of working functionality
    arc_weights = r".\weights\best.pt"
    im_path = r".\image.png"
    ##
    run(weights=arc_weights,
        source=im_path,
        nosave=False,
        agnostic_nms=True,
        iou_thres=0.3,
        conf_thres=0.2,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=r".\YOLO",
        imgsz=(640,640),
        name=f"test",
        exist_ok=True,
        skip_existing=True,
        image_sort=sorter)