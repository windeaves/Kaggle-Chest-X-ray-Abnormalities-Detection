import json

def rf(f, x):
    if isinstance(x, list):
        for xs in x:
            rf(f, xs)
    else:
        f.write(str(x) + " ")

def toSub(jsonobj, filename):
    f = open(filename, "w")
    f.write("image_id,PredictionString\n")
    for xs in jsonobj:
        for x in xs.keys():
            f.write(x)
            f.write(",")
            for b in xs[x]:
                rf(f, b["cls"] - 1)
                rf(f, b["conf"])
                rf(f, b["bbox"])
            f.write("\n")

def cvt(line):
    intf = lambda x : [int(xs) for xs in x]

    line = line.strip("\n")
    line = line.split(",")
    
    img_name = line[0]
    
    boxes = list()

    line = line[1].split(" ")
    for a in line:
        if a == "":
            line.remove(a)

    for i in range(len(line) // 6):
        if(int(line[i*6]) == 14):
            continue
        boxes.append(dict(cls=int(line[i*6]) + 1, conf=float(line[i*6+1]), bbox=intf(line[i * 6 + 2:(i+1) * 6])))
    return { img_name : boxes }
            


def stojson(filename):
    jsonobj = list()
    with open(filename, "r") as f:
        for x in f.readlines():
            jsonobj.append(cvt(x))
    with open("trs" + ".json", "w") as f:
        json.dump(jsonobj, f)