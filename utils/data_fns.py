import os
import codecs
import ujson
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils.data_utils import Vocabulary, TransVocabulary

np.random.seed(12345)
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}

PAD, UNK, SP = "<PAD>", "<UNK>", "<SP>"
emb_path = os.path.join(os.path.expanduser('~'), "utilities", "embeddings")


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def iob_to_iobes(labels):
    """IOB -> IOBES"""
    iob_to_iob2(labels)
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iob_to_iob2(labels):
    """Check that tags have a valid IOB format. Tags in IOB1 format are converted to IOB2."""
    for i, tag in enumerate(labels):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or labels[i - 1] == 'O':  # conversion IOB1 to IOB2
            labels[i] = 'B' + tag[1:]
        elif labels[i - 1][1:] == tag[1:]:
            continue
        else:
            labels[i] = 'B' + tag[1:]
    return True


def word_convert(word, word_lower=True, char_lower=False):
    if char_lower:
        char = [c for c in word.lower()]
    else:
        char = [c for c in word]
    if word_lower:
        word = word.lower()
    return word, char


def raw_dataset_iter(filename, word_lower=True, char_lower=False):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, chars, labels = [], [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 and len(words) != 0:
                yield words, chars, labels
                words, chars, labels = [], [], []
            else:
                word, label = line.split("\t")
                word, char = word_convert(word, word_lower=word_lower, char_lower=char_lower)
                words.append(word)
                chars.append(char)
                labels.append(label)
        if len(words) != 0:
            yield words, chars, labels


def load_dataset(filename, task, iobes, word_lower=True, char_lower=False):
    dataset = list()
    if not os.path.exists(filename):
        return dataset
    for words, chars, labels in raw_dataset_iter(filename, word_lower, char_lower):
        if iobes and "pos" not in task:
            labels = iob_to_iobes(labels)
        dataset.append({"words": words, "chars": chars, "labels": labels})
    return dataset


def load_emb_vocab(data_path, dim):
    vocab = list()
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_emb(word_dict, data_path, dim):
    vectors = np.zeros([len(word_dict), dim])
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vectors"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_idx = word_dict[word]
                vectors[word_idx] = np.asarray(vector)
    return np.asarray(vectors)


def build_token_counters(datasets):
    word_counter = Counter()
    char_counter = Counter()
    label_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                word_counter[word] += 1
            for chars in record["chars"]:
                for char in chars:
                    char_counter[char] += 1
            for label in record["labels"]:
                label_counter[label] += 1
    return word_counter, char_counter, label_counter


def build_dataset(data, word_dict, char_dict, label_dict):
    dataset = []
    for record in data:
        chars_list, words = [], []
        for word in record["words"]:
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        for char in record["chars"]:
            chars = [char_dict[c] if c in char_dict else char_dict[UNK] for c in char]
            chars_list.append(chars)
        labels = [label_dict[label] for label in record["labels"]]
        dataset.append({"words": words, "words_str": record["words"], "chars": chars_list, "labels": labels})
    return dataset


def read_data_and_vocab(task, config):
    path = "datasets/{}/".format(task)
    train_data = load_dataset(path + "train.txt", task, config.iobes, config.word_lower, config.char_lower)
    dev_data = load_dataset(path + "valid.txt", task, config.iobes, config.word_lower, config.char_lower)
    test_data = load_dataset(path + "test.txt", task, config.iobes, config.word_lower, config.char_lower)
    word_counter, char_counter, label_counter = build_token_counters([train_data, dev_data, test_data])
    return train_data, dev_data, test_data, word_counter, char_counter, label_counter


def write_to_jsons(datasets, files, save_path):
    for dataset, file in zip(datasets, files):
        write_json(os.path.join(save_path, file), dataset)


def create_token_weight(token_dict, token_vocab, token_counter):
    token_count = dict()
    for token, count in token_counter.most_common():
        if token in token_dict:
            token_count[token] = token_count.get(token, 0) + count
        else:
            token_count[UNK] = token_count.get(UNK, 0) + count
    sum_count = float(sum(token_count.values()))
    token_weight = [float(token_count[token]) / sum_count for token in token_vocab[1:]]  # exclude PAD
    return token_weight


def process_word_counter(word_counter, wordvec_path, word_dim, threshold=0):
    if wordvec_path is not None:
        wordvec_path = os.path.join(emb_path, wordvec_path)
    if wordvec_path is not None and os.path.exists(wordvec_path):
        emb_vocab = load_emb_vocab(wordvec_path, dim=word_dim)
        word_vocab = list()
        for word, _ in word_counter.most_common():
            if word in emb_vocab:
                word_vocab.append(word)
        tmp_word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        vectors = filter_emb(tmp_word_dict, wordvec_path, word_dim)
        word_vocab = [PAD, UNK] + word_vocab
        word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        word_weight = create_token_weight(word_dict, word_vocab, word_counter)
    else:
        word_vocab = [PAD, UNK] + [word for word, count in word_counter.most_common() if count >= threshold]
        word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        vectors = None
        word_weight = create_token_weight(word_dict, word_vocab, word_counter)
    return word_dict, vectors, word_weight


def process_char_counter(char_counter, threshold):
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= threshold]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    char_weight = create_token_weight(char_dict, char_vocab, char_counter)
    return char_dict, char_weight


def process_label_counter(label_counter):
    label_vocab = ["O"] + [label for label, _ in label_counter.most_common() if label != "O"]
    label_dict = dict([(label, idx) for idx, label in enumerate(label_vocab)])
    return label_dict


def process_base(cfg):
    train_data, dev_data, test_data, word_counter, char_counter, label_counter = read_data_and_vocab(cfg.task, cfg)
    # build word vocab
    word_dict, vectors, word_weight = process_word_counter(word_counter, cfg.wordvec_path, cfg.word_dim,
                                                           cfg.word_threshold)
    # build char vocab
    char_dict, char_weight = process_char_counter(char_counter, cfg.char_threshold)
    # build label vocab
    label_dict = process_label_counter(label_counter)
    # create indices dataset
    if cfg.use_dev:
        train_data = train_data + dev_data
    train_set = build_dataset(train_data, word_dict, char_dict, label_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, label_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, label_dict)
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "label_dict": label_dict, "vectors": vectors,
             "word_weight": word_weight, "char_weight": char_weight}
    return {"train": train_set, "dev": dev_set, "test": test_set}, Vocabulary(vocab)


def process_transfer(cfg):
    s_train_data, s_dev_data, s_test_data, s_word_counter, s_char_counter, s_label_counter = read_data_and_vocab(
        cfg.src_task, config=cfg)
    t_train_data, t_dev_data, t_test_data, t_word_counter, t_char_counter, t_label_counter = read_data_and_vocab(
        cfg.tgt_task, config=cfg)
    # build word vocab
    if cfg.share_word:
        s_word_counter = s_word_counter + t_word_counter
        word_dict, vectors, word_weight = process_word_counter(s_word_counter, cfg.src_wordvec_path, cfg.src_word_dim,
                                                               cfg.word_threshold)
        src_word_dict, tgt_word_dict = word_dict.copy(), word_dict.copy()
        src_vectors, tgt_vectors = vectors, None
        src_word_weight, tgt_word_weight = word_weight, None
    else:
        src_word_dict, src_vectors, src_word_weight = process_word_counter(s_word_counter, cfg.src_wordvec_path,
                                                                           cfg.src_word_dim, cfg.word_threshold)
        tgt_word_dict, tgt_vectors, tgt_word_weight = process_word_counter(t_word_counter, cfg.tgt_wordvec_path,
                                                                           cfg.tgt_word_dim, cfg.word_threshold)
    # build char vocab
    s_char_counter = s_char_counter + t_char_counter
    char_dict, char_weight = process_char_counter(s_char_counter, cfg.char_threshold)
    # build label vocab
    if cfg.share_label:
        s_label_counter = s_label_counter + t_label_counter
        label_dict = process_label_counter(s_label_counter)
        src_label_dict = label_dict.copy()
        tgt_label_dict = label_dict.copy()
    else:
        src_label_dict = process_label_counter(s_label_counter)
        tgt_label_dict = process_label_counter(t_label_counter)
    # create indices dataset
    src_train_set = build_dataset(s_train_data, src_word_dict, char_dict, src_label_dict)
    src_dev_set = build_dataset(s_dev_data, src_word_dict, char_dict, src_label_dict)
    src_test_set = build_dataset(s_test_data, src_word_dict, char_dict, src_label_dict)
    if cfg.use_dev:
        t_train_data = t_train_data + t_dev_data
    tgt_train_set = build_dataset(t_train_data, tgt_word_dict, char_dict, tgt_label_dict)
    tgt_dev_set = build_dataset(t_dev_data, tgt_word_dict, char_dict, tgt_label_dict)
    tgt_test_set = build_dataset(t_test_data, tgt_word_dict, char_dict, tgt_label_dict)
    vocab = {"src_word_dict": src_word_dict, "src_vectors": src_vectors, "src_word_weight": src_word_weight,
             "tgt_word_dict": tgt_word_dict, "tgt_vectors": tgt_vectors, "tgt_word_weight": tgt_word_weight,
             "char_dict": char_dict, "char_weight": char_weight, "src_label_dict": src_label_dict,
             "tgt_label_dict": tgt_label_dict}
    return {"train": src_train_set, "dev": src_dev_set, "test": src_test_set}, \
           {"train": tgt_train_set, "dev": tgt_dev_set, "test": tgt_test_set}, TransVocabulary(vocab)
