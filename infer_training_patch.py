import tifffile
from torch_em.util.prediction import predict_with_halo

from inference_hpc import sigmoid_postprocess
from model_hpc import load_trained_model
model_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_37227920/checkpoints/best_model.pth"
config_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_36471054/logs/config.json"
volume_path = "/home/asvetlove/data/segmentation/inference_examples/POR_20to200_20231022_AM_01_epo_02/"
model = load_trained_model(model_path, config_path)
model.eval()
patch = tifffile.imread('/home/asvetlove/PycharmProjects/TREC_seg_unet/data/ml_patches/POR_20o200_20231022_AM_01_epo_01/patch_0002.tif')

prediction = predict_with_halo(
    patch,
    model,
    gpu_ids=[0],
    block_shape= [128,128,128],
    halo=[64,64,64],
    mask=None,
    postprocess=sigmoid_postprocess,
    preprocess=None
)


foreground, boundaries = prediction[0], prediction[1]
tifffile.imwrite('/home/asvetlove/data/segmentation/training_test_infb.tiff', boundaries)
tifffile.imwrite('/home/asvetlove/data/segmentation/training_test_inff.tiff', foreground)