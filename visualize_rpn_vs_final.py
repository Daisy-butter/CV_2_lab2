import torch
import os
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector
from mmcv import imread

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


def get_data_sample(cfg, model, image_path):
    # 构建测试数据
    data = dict(img_path=image_path)
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = pipeline(data)
    collated_data = default_collate([data])
    data = model.data_preprocessor(collated_data, False)
    return data, data['inputs'][0], data['data_samples'][0]


def visualize_predictions(cfg, model, image_path, out_dir):
    # 获取数据
    data, image_tensor, data_sample = get_data_sample(cfg, model, image_path)

    # 执行推理，获取 proposal 和 final prediction
    with torch.no_grad():
        x = model.extract_feat(data['inputs'])
        batch_data_samples = [data_sample]

        # 获取RPN proposals
        proposals = model.rpn_head.predict(
            x, batch_data_samples, rescale=True
        )[0]  # 只取当前图像的proposals

        # 获取Final predictions
        final_predictions = model.roi_head.predict(
            x, [proposals], batch_data_samples, rescale=True
        )[0]  # 同样只处理当前图像

    # 构造 visualizer（加fallback配置）
    visualizer_cfg = cfg.get('visualizer', dict(type='DetLocalVisualizer'))
    visualizer_cfg['name'] = 'visualizer_rpn_vs_final'
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.dataset_meta = model.dataset_meta

    # 图像转 uint8 RGB 格式
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

    # 可视化 Proposal Boxes
        # 可视化 Proposal Boxes
    proposal_sample = DetDataSample()
    num_proposals = proposals.bboxes.shape[0]
    proposal_sample.pred_instances = InstanceData(
        bboxes=proposals.bboxes,  # 获取 proposal 框
        scores=proposals.scores,  # proposal 的置信度分数
        labels=torch.zeros(num_proposals, dtype=torch.long)  # 默认类标签索引为0，即“未知”
    )
    proposal_sample.set_metainfo(data_sample.metainfo)

    visualizer.add_datasample(
        name='proposal_boxes',
        image=img_np,
        data_sample=proposal_sample,
        draw_gt=False,
        draw_pred=True,
        pred_score_thr=0.0,  # 显示所有proposal
        out_file=os.path.join(out_dir, 'proposal_boxes.jpg')
    )


    # 可视化 Final Prediction
    if hasattr(final_predictions, 'masks') and final_predictions.masks is not None:
        masks = final_predictions.masks
        if len(masks) > 0:
            resized_masks = []
            for mask in masks:  # 遍历每个 mask，并调整大小
                resized_mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :],  # 添加 batch 和 channel 维度
                    size=img_np.shape[:2],  # 调整到与输入图像相同的 (height, width)
                    mode='bilinear',
                    align_corners=False
                )[0, 0]  # 移除 batch 和 channel 维度
                resized_masks.append(resized_mask)
            final_predictions.masks = torch.stack(resized_masks)  # 合并调整后的 mask
        else:
            # 若无有效 masks，则设置为空张量
            final_predictions.masks = torch.empty((0, *img_np.shape[:2]), dtype=torch.uint8)
    else:
    # 若不存在 masks 属性，也设置为空张量
        final_predictions.masks = torch.empty((0, *img_np.shape[:2]), dtype=torch.uint8)




    final_sample = DetDataSample()
    final_sample.pred_instances = InstanceData(
        bboxes=final_predictions.bboxes,  # 直接获取 final_predictions 的 bboxes
        scores=final_predictions.scores,  # 直接获取 final_predictions 的 scores
        labels=final_predictions.labels,  # 直接获取 final_predictions 的 labels
        masks=final_predictions.masks  # 若无 masks，则为 None
    )
    final_sample.set_metainfo(data_sample.metainfo)

    visualizer.add_datasample(
        name='final_predictions',
        image=img_np,
        data_sample=final_sample,
        draw_gt=False,
        draw_pred=True,
        pred_score_thr=0.5,  # 可以调整阈值，仅显示高置信度预测
        out_file=os.path.join(out_dir, 'final_predictions.jpg')
    )



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize RPN vs Final prediction')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--weights', help='Path to checkpoint file')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--out-dir', help='Directory to save results')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    # 初始化配置和模型
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    register_all_modules()
    model = init_detector(cfg, args.weights, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    visualize_predictions(cfg, model, args.image, args.out_dir)
