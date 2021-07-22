import os

DATABASE_PATH = "Detection"
IMAGES_PATH = DATABASE_PATH
ANNOTS_PATH = os.path.sep.join([DATABASE_PATH, "gt.txt"])

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INIT_LR = 1e-5
NUM_EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 50