import time
import torch
import torch.nn as nn
import torchvision
from utils.common import *


def make_anchors(imh, imw, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = imh // stride, imw // stride
        sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset  # shift y
        sy, sx = meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class YOLOv8PostProcess:
    def __init__(self, imh, imw, conf_thres, iou_thres, nc=17):
        self.is_first = True
        self.nc = nc  # number of classes
        self.nl = None  # number of detection layers
        self.imh = imh
        self.imw = imw
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reg_max = (
            16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        )
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.dfl = DFL(self.reg_max)

    def __call__(self, npu_outs):
        npu_outs = self.rearrange_npu_out(npu_outs)
        npu_outs = self.decode(npu_outs)
        npu_outs = self.non_max_suppression(npu_outs, self.conf_thres, self.iou_thres)
        if npu_outs[0].nelement() == 0:
            return None
        return npu_outs
    
    # repaired rearrange_npu_out
    def rearrange_npu_out(self, x):
        """
        x: 리스트 길이 6
        [ (20,20,64), (20,20,80),
            (40,40,64), (40,40,80),
            (80,80,64), (80,80,80) ]
        형태의 numpy 배열(HWC)
        """

        # 1) 스케일별 (box_idx, cls_idx, h, w) 묶음 만들기
        scales = []  # list of dicts: {"h":h, "w":w, "box_i":idx64, "cls_i":idx80}
        used = [False] * len(x)
        for i in range(0, len(x), 2):
            h0, w0, c0 = x[i].shape
            h1, w1, c1 = x[i+1].shape
            # 채널 확인
            
            box_ch = self.reg_max * 4
            cls_ch = self.nc
            
            if c0 == box_ch and c1 == cls_ch and (h0, w0) == (h1, w1):
                scales.append({"h": h0, "w": w0, "box_i": i,   "cls_i": i+1})
            elif c0 == cls_ch and c1 == box_ch and (h0, w0) == (h1, w1):
                scales.append({"h": h0, "w": w0, "box_i": i+1, "cls_i": i})
            else:
                raise ValueError(f"Unexpected pair: {x[i].shape} & {x[i+1].shape}")

        # 2) 스케일을 큰 H*W 순(= stride 8 → 16 → 32)으로 정렬
        scales.sort(key=lambda d: d["h"] * d["w"], reverse=True)

        # 3) 최초 1회, 앵커/stride를 입력 스케일에 맞춰 동적으로 구성
        if self.is_first:
            self.is_first = False
            self.nl = len(scales)
            # stride = imh / h  (예: 640/80=8, 640/40=16, 640/20=32)
            self.stride = [self.imh // s["h"] for s in scales]
            # 앵커는 위 순서대로 생성해야 함
            self.anchors, self.strides = (
                xx.transpose(0, 1)
                for xx in make_anchors(self.imh, self.imw, self.stride, 0.5)
            )

        # 4) 두 분기(DFL 64, CLS 80)를 채널 방향으로 합쳐 (B, no, h, w)
        y = []
        for s in scales:
            h, w = s["h"], s["w"]

            box = torch.from_numpy(x[s["box_i"]])  # (h, w, 64)
            cls = torch.from_numpy(x[s["cls_i"]])  # (h, w, 80)

            if box.dim() == 3: box = box.unsqueeze(0)  # (1,h,w,64)
            if cls.dim() == 3: cls = cls.unsqueeze(0)  # (1,h,w,80)

            box = box.permute(0, 3, 1, 2).contiguous()  # (B,64,h,w)
            cls = cls.permute(0, 3, 1, 2).contiguous()  # (B,80,h,w)

            tmp = torch.cat([box, cls], dim=1).contiguous()  # (B, 64+80, h, w)
            y.append(tmp)

        return y


    def decode(self, x):
        box, cls = torch.cat([xi.flatten(start_dim=2) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(
            prediction, (list, tuple)
        ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = "mps" in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = self.nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            box = xywh2xyxy(
                box
            )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1
                )
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[
                    conf.view(-1) > conf_thres
                ]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        return output