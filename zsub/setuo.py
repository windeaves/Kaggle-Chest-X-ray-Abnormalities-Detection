from ensemble_boxes import *
import json
from PIL import Image
import os

def read_result(filename):
    rjson = dict()
    r = dict()
    with open(filename, "r") as f:
        rjson = json.load(f)
    for rj in rjson:
        for key in rj.keys():
            r[key.strip(',')] = rj[key]
    return r

def N(box, ims):
    for i in range(len(box)):
        box[i] /= ims[i % 2]
    return box

def deN(box, ims):
    for i in range(len(box)):
        box[i] = int(box[i] * ims[i  % 2])
    return box

def defloat(x):
    r = list()
    for xs in x:
        r.append(float(xs))
    return r

def fusion(r1, r2, img_size, iou_thr, skip_box_thr):
    getbox = lambda x : [N(m["bbox"], img_size) for m in x]
    getscore = lambda x : [m["conf"] for m in x]
    getcls = lambda x : [m["cls"] for m in x]

    bp = lambda f, x1, x2 : [f(x1), f(x2)]

    bbox = bp(getbox, r1, r2)
    score = bp(getscore, r1, r2)
    label = bp(getcls, r1, r2)
    weights = [1, 1]
    
    boxes, scores, labels = weighted_boxes_fusion(bbox, score, label, weights, iou_thr, skip_box_thr)
    
    return [{"bbox": deN(boxes[i], img_size).tolist(), "conf": float(scores[i]), "cls": int(labels[i])} for i in range(len(boxes))]

def pic_size():
    root = "D:\\kaggle\\chest-x-ray\\dataset\\images\\test"
    img_lists = os.listdir(root)
    r = dict()
    for img in img_lists:
        r[img.replace(".jpeg", "")] = (Image.open(os.path.join(root, img)).size)
    return r


if __name__ == "__main__":
    img_size = pic_size()

    name_1 = "result250.json"
    name_2 = "result300.json"
    iou_thr = 0.25
    skip_box_thr = 0.0001
    r1 = read_result(name_1)
    r2 = read_result(name_2)
    
    keys = set(r1.keys()) 
    keys.intersection_update(set(r2.keys()))
    print([len(r1.keys()), len(r2.keys()), len(keys)])

    result = list()
    
    for key in keys:
        result.append({key : fusion(r1[key], r2[key], img_size[key], iou_thr, skip_box_thr)})

    save_file_name = name_1.replace(".json", "") + "_" +name_2.replace(".json", "") + "_" + str(iou_thr) + "_" + str(skip_box_thr) + ".json"
    with open(save_file_name, "w") as f:
        json.dump(result, f)
    import json2sub
    json2sub.toSub(result, save_file_name.replace(".json", ".txt"))
    
def fusion_score(model1, model2):
    mini = min(model1.score, model2.score)
    maxi = max(model1.score, model2.score)
    return random(mini, maxi)