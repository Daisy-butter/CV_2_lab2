import os
import json
import random
import shutil
from pathlib import Path

# 固定随机种子
random.seed(42)

# 路径配置（请根据你本地环境修改）
SRC_JSON = 'data/coco_converted/voc07_trainval.json'  # VOC2007 转换后的 COCO JSON
SRC_IMG_DIR = 'data/VOCdevkit/VOC2007/JPEGImages'     # 原始图片所在路径
OUT_DIR = 'data/coco'                                 # 输出 COCO 风格的数据集路径

# 创建 COCO 所需的目录结构
for folder in ['annotations', 'train2017', 'val2017', 'test2017']:
    Path(os.path.join(OUT_DIR, folder)).mkdir(parents=True, exist_ok=True)

# 加载 COCO 标注文件
with open(SRC_JSON, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# 打乱并按 8:1:1 划分
random.shuffle(images)
n = len(images)
train_imgs = images[:int(0.8 * n)]
val_imgs = images[int(0.8 * n):int(0.9 * n)]
test_imgs = images[int(0.9 * n):]

# 构建 image_id -> anns 映射
image_id_to_anns = {}
for ann in annotations:
    image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

# 子集保存函数
def save_subset(images, name):
    image_ids = {img['id'] for img in images}
    anns = [ann for ann in annotations if ann['image_id'] in image_ids]
    
    # 更新每个 image 的 file_name 为纯文件名
    for img in images:
        img['file_name'] = os.path.basename(img['file_name'])

    # 保存 json 文件
    out_json = {
        'images': images,
        'annotations': anns,
        'categories': categories
    }
    with open(os.path.join(OUT_DIR, 'annotations', f'instances_{name}.json'), 'w') as f:
        json.dump(out_json, f)

    # 拷贝图像文件
    for img in images:
        filename = img['file_name']
        src_path = os.path.join(SRC_IMG_DIR, filename)
        dst_path = os.path.join(OUT_DIR, name, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"❌ Warning: Image file not found: {src_path}")

# 执行划分并保存
save_subset(train_imgs, 'train2017')
save_subset(val_imgs, 'val2017')
save_subset(test_imgs, 'test2017')

print("✅ VOC2007 数据已成功划分为 COCO 格式的 train/val/test 子集。")
