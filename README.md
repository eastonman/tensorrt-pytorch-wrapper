# tensorrt-pytorch-wrapper
A wrapper makes TensorRT engine accept PyTorch Cuda Tensor.


## Usage

### Install Dependencies
- PyCuda
- PyTorch (tested with 1.6.0 & 1.9.0)
- TensorRT (Python package)

### Install
Copy `engine.py` into your project directory

### Example

```
from engine import Engin
class Rcnn(nn.Module):
    def __init__(self, backbone, det_classes, seg_classes, **kwargs):
        super(Rcnn_Deeplab, self).__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        del self.backbone
        self.mask_feat_head = MaskFeatHead(
            out_channels, 4) if seg_classes else None
        self.seg_head = SegmentHead(
            out_channels, seg_classes) if seg_classes else None
        self.det_head = FasterRCNN(out_channels, det_classes, **kwargs)
        self.seg_classes, self.focal_loss = seg_classes, FocalLoss(
            reduction='sum')
        self.mixup, self.seg_classes = data_mixup(), seg_classes

        # Intermediate
        self.feats = []
        self.feats.append(torch.zeros((8, 256, 32, 32),
                          dtype=torch.float16, device='cuda'))
        self.feats.append(torch.zeros((8, 256, 64, 64),
                          dtype=torch.float16, device='cuda'))
        self.feats.append(torch.zeros(
            (8, 256, 128, 128), dtype=torch.float16, device='cuda'))
        self.feats.append(torch.zeros(
            (8, 256, 256, 256), dtype=torch.float16, device='cuda'))
        self.feats.append(torch.zeros((8, 256, 16, 16),
                          dtype=torch.float16, device='cuda'))
        # Tensorrt
        self.engine = Engine(max_batch_size=8, outputs=self.feats, onnx_file_path="backbone.onnx",
                             engine_file_path="model/resnet101_fpn.trt", fp16_mode=True)


	def forward(self, images):
		images = images.to(memory_format=torch.contiguous_format)
            self.engine.gpu_forward([images])
		...
```