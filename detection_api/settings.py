classes = 2
dataset_root = "/home/super/Projects/dataset/bbox/custom"
dataset_prefix = "/home/super/Projects/dataset/bbox" 
train = dataset_root + "/train.txt"
valid = dataset_root + "/valid.txt"
names = dataset_root + "/classes.names"

class_names = ["face", "car_plate"]

epochs = 200
batch_size = 12
gradient_accumulations = 1
model_def = "model/yolov3-custom.cfg"
pretrained_weights = "checkpoints/yolov3_ckpt_31.pth"
n_cpu = 16
img_size = 704
checkpoint_interval = 1
evaluation_interval = 1
compute_map = True
multiscale_training = True
start_epoch = 32
device = "cuda"
lr = 0.0005