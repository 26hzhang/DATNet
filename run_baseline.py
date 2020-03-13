import os
from utils.data_fns import process_base
from utils.data_utils import Dataset, boolean_string
from models.baseline import BaseModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--mode", type=int, default=0, help="fix to 0 for basic settings")
parser.add_argument("--use_gpu", type=boolean_string, default=True, help="if use GPU for training")
parser.add_argument("--gpu_idx", type=str, default="0", help="indicate which GPU is used for training and inference")
parser.add_argument("--random_seed", type=int, default=86, help="random seed")
parser.add_argument("--train", type=boolean_string, default=True, help="if True, train the model")
parser.add_argument("--elmo", type=boolean_string, default=False, help="use ELMo embeddings")
parser.add_argument("--restore_model", type=boolean_string, default=False, help="restore pre-trained and fine-tune")
parser.add_argument("--at", type=boolean_string, default=False, help="if True, use adversarial training")
parser.add_argument("--norm_emb", type=boolean_string, default=False, help="if True, normalizing embeddings for AT")
parser.add_argument("--task", default="conll03_en_ner", help="specify the task")
parser.add_argument("--iobes", type=boolean_string, default=True, help="if True, use IOBES scheme, otherwise, IOB2")
parser.add_argument("--use_dev", type=boolean_string, default=False, help="use development dataset for training")
parser.add_argument("--train_rate", default="1.0", help="training dataset ratio")
parser.add_argument("--word_lower", type=boolean_string, default=False, help="lowercase words")
parser.add_argument("--char_lower", type=boolean_string, default=False, help="lowercase characters")
parser.add_argument("--word_threshold", type=int, default=5, help="character threshold")
parser.add_argument("--char_threshold", type=int, default=5, help="character threshold")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--wordvec_path", type=str, default="glove.840B.300d.txt", help="embedding dim must equal word dim")
parser.add_argument("--word_project", type=boolean_string, default=False, help="word project")
parser.add_argument("--char_dim", type=int, default=100, help="character embedding dimension")
parser.add_argument("--tune_emb", type=boolean_string, default=False, help="optimizing word embeddings while training")
parser.add_argument("--hw_layer", type=int, default=2, help="number of highway layers used")
parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="CNN kernels for char")
parser.add_argument("--filters", type=int, nargs="+", default=[10, 20, 30, 40, 50], help="CNN features for char")
parser.add_argument("--concat_rnn", type=boolean_string, default=True, help="concatenate outputs of bi-rnn layer")
parser.add_argument("--num_units", type=int, default=200, help="number of units for RNN")
parser.add_argument("--epsilon", type=float, default=5.0, help="epsilon")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_decay", type=float, default=0.05, help="learning rate decay factor")
parser.add_argument("--decay_step", type=int, default=10, help="learning rate decay steps")
parser.add_argument("--optimizer", type=str, default="lazyadam", help="optimizer: [rmsprop | adadelta | adam | ...]")
parser.add_argument("--grad_clip", type=float, default=5.0, help="maximal gradient norm")
parser.add_argument("--epochs", type=int, default=50, help="train epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--emb_drop_rate", type=float, default=0.2, help="dropout rate for embeddings")
parser.add_argument("--rnn_drop_rate", type=float, default=0.5, help="dropout rate for embeddings")
parser.add_argument("--model_name", type=str, default="base_model", help="model name")
parser.add_argument("--max_to_keep", type=int, default=1, help="maximum trained model to be saved")
parser.add_argument("--no_imprv_tolerance", type=int, default=None, help="no improvement tolerance")
config = parser.parse_args()

# os environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx


print("load dataset...")
datasets, vocab = process_base(config)
train_rate = int(config.train_rate) if float(config.train_rate) > 1.0 else float(config.train_rate)
dataset = Dataset(datasets, batch_size=config.batch_size, train_rate=train_rate, shuffle=True)

print("build model and train...")
model = BaseModel(config, vocab)
if config.restore_model:
    model.restore_last_session()
if config.train:
    model.train(dataset)
model.restore_last_session()
model.evaluate_data(dataset.test_batches(), name="test")
model.close_session()
