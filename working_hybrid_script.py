import numpy as np
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
from membrane.models import fgru_tmp as fgru


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, basestring):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


SHAPE = np.array([128, 128, 128])
SEED = np.array([5, 5, 5])
MEMBRANE_MODEL = 'fgru_tmp'  # Allow for dynamic import
MEMBRANE_CKPT = '/gpfs/data/tserre/data/connectomics/checkpoints/global_2_fb_wide_mini_fb_hgru3d_berson_0_berson_0_2018_10_29_14_58_16_883649/model_63000.ckpt-63000'
CONF = [4992, 16000, 10112]
PATH_STR = '/local1/dlinsley/connectomics/mag1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'
MEM_STR = '/local1/dlinsley/connectomics/mag1/x%s/y%s/z%s/membrane_110629_k0725_mag1_x%s_y%s_z%s.npy'
PATH_EXTENT = 2

# 1. select a volume
if PATH_EXTENT == 1:
    path = PATH_STR % (
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4),
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4))
    vol = np.fromfile(path, dtype='uint8').reshape(SHAPE)

else:
    vol = np.zeros((np.array(SHAPE) * PATH_EXTENT))
    for x in range(PATH_EXTENT):
        for y in range(PATH_EXTENT):
            for z in range(PATH_EXTENT):
                path = PATH_STR % (
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4),
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4))
                v = np.fromfile(path, dtype='uint8').reshape(SHAPE)
                vol[
                    x * SHAPE[0] : x * SHAPE[0] + SHAPE[0],
                    y * SHAPE[1] : y * SHAPE[1] + SHAPE[1],
                    z * SHAPE[2] : z * SHAPE[2] + SHAPE[2]] = v
vol = vol.transpose(2, 1, 0) / 255.

# 2. Predict its membranes
membranes = fgru.main(
    test=vol,
    evaluate=True,
    adabn=True,
    test_input_shape=np.concatenate((SHAPE * PATH_EXTENT, [1])).tolist(),
    test_label_shape=np.concatenate((SHAPE * PATH_EXTENT, [12])).tolist(),
    checkpoint=MEMBRANE_CKPT)

# 3. Concat the volume w/ membranes and pass to FFN
membranes = 1 - np.stack((vol, membranes[0, :, :, :, :3].mean(-1) > 0.5), axis=-1)
mpath = MEM_STR % (
    pad_zeros(SEED[0], 4),
    pad_zeros(SEED[1], 4),
    pad_zeros(SEED[2], 4),
    pad_zeros(SEED[0], 4),
    pad_zeros(SEED[1], 4),
    pad_zeros(SEED[2], 4))
np.save(mpath, membranes)
config = '''image {
  hdf5: "%s"}
image_mean: 154
image_stddev: 33
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/gpfs/data/tserre/data/ffn_ckpts/model.ckpt-255052"
model_name: "convstack_3d_bn_f.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 12, \\"fov_size\\": [57, 57, 13], \\"deltas\\": [8, 8, 3]}" 
segmentation_output_dir: "/gpfs/data/tserre/data/ding_segmentations"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.5
  min_boundary_dist { x: 1 y: 1 z: 1}
  segment_threshold: 0.3
  min_segment_size: 100
}''' % mpath

req = inference_pb2.InferenceRequest()
_ = text_format.Parse(config, req)

from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
req = inference_pb2.InferenceRequest()
_ = text_format.Parse(config, req)
runner = inference.Runner()
runner.start(req)
canvas, alignment = runner.make_canvas((0, 0, 0), (250, 250, 250))
canvas, alignment = runner.make_canvas((0, 0, 0), (128, 128, 128))
canvas.segment_at((60, 60, 60),  # zyx
                  dynamic_image=inference.DynamicImage(),
                  vis_update_every=1)

