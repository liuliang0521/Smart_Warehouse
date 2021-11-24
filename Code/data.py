import argparse
import time
from pathlib import Path
import os
import shutil
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default=r'.\runs\train\exp2\weights\best.pt',help='model.pt path(s)')
parser.add_argument('--source', type=str, default=r'.\pic', help='source')  # 修改成自己的测试图片所在了路径
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()
print(opt)
check_requirements(exclude=('pycocotools', 'thop'))

def detect(path):
    """
    source：图像的路径
    weight：权重文件路径
    view_img：False
    save_txt：False
    imgsz：图像大小
    """
    ans = []
    read_path_need=""
    save_path_need=""
    pic_name=""
    if path == "NULL":
        if os.path.exists("./pic/picture.jpg"):
            os.remove("./pic/picture.jpg")
        if os.path.exists("./answer/picture.jpg"):
            os.remove("./answer/picture.jpg")
        cap = cv2.VideoCapture(1)
        # 从摄像头获取图像，第一个为布尔变量表示成功与否，第二个变量是图像
        ret, filename = cap.read()
        # 保存图像至Haar相同路径
        cv2.imwrite('./pic/picture.jpg', filename)
        # 释放摄像头资源
        cap.release()
        read_path_need = "./pic/picture.jpg"
        save_path_need = "./answer/picture.jpg"
        pic_name="picture.jpg"
    else:
        if os.path.exists("./answer/pic ("+str(int(path)+1)+").jpg"):
            os.remove("./answer/pic ("+str(int(path)+1)+").jpg")
        read_path_need = "./pic/pic ("+str(int(path)+1)+").jpg"
        save_path_need = "./answer/pic ("+str(int(path)+1)+").jpg"
        pic_name="pic ("+str(int(path)+1)+").jpg"
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = True

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(read_path_need, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        counterMap1 = {}  # added
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = save_path_need  # img.jpg
            txt_path = str("./result/answer.txt")  # img.txt
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    clsName = names[int(cls)]
                    if clsName not in counterMap1.keys():
                        counterMap1[clsName] = 1
                    else:
                        counterMap1[clsName] += 1
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            need = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            number = ["CA001", "CA002", "CA003", "CA004", "CD001", "CD002", "CD003", "CD004", "CD005", "CD006", "ZA001",
                      "ZA002", "ZA003", "ZA004", "ZA005", "ZA006", "ZB001", "ZB002", "ZB003", "ZB004", "ZB005", "ZB006",
                      "ZB007", "ZB008", "ZB009", "ZB010", "ZC001", "ZC002", "ZC003", "ZC004", "ZC005", "ZC006", "ZC007",
                      "ZC008", "ZC009", "ZC010", "ZC011", "ZC012", "ZC013", "ZC014", "ZC015", "ZC016", "ZC017", "ZC018",
                      "ZC019", "ZC020", "ZC021", "ZC022", "ZC023"]
            with open(txt_path,"w") as f:
                for index,item in enumerate(counterMap1.items()):
                    for i in range(0,len(number)):
                        if(item[0]==number[i]):
                            need[i] = item[1]
                            break
            ans = need
            if True:
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
    im = Image.open(read_path_need)
    im.show()
    im = Image.open(save_path_need)
    im.show()
    name = ["抽纸", "卷纸", "牙刷", "胶带", "苹果", "水晶梨", "哈密瓜", "猕猴桃", "柚子", "香蕉", "香皂", "洗手液", "牙膏", "花露水", "洗澡鸭",
            "文具盒", "八宝粥", "老干妈", "曲奇饼", "甜橙汁", "口香糖", "公仔面", "苏打饼干", "薯片", "薯条", "香瓜子", "雪碧", "可乐", "芬达", "红牛",
            "AD钙奶", "果粒橙", "王老吉", "加多宝", "冰红茶", "绿茶", "冰糖雪梨", "茶派", "椰汁", "农夫山泉", "哇哈哈", "百岁山", "怡宝", "恒大冰泉",
            "康师傅", "今麦郎", "昆仑山", "雀巢优活", "冰露"]
    for i in range(0,len(ans)):
        if(ans[i]!=0) :
            print(name[i]+"有："+str(ans[i])+"个")

    return ans



def function(path):
    #调用函数
    ans = []
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                ans = detect(path)
                strip_optimizer(opt.weights)
        else:
            ans =detect(path)
    print(ans)
    return ans

import numpy as np
data = []
for i in range(0,5):
    data.append(function(str(i)))
print(data)
np.save("data.npy", data)