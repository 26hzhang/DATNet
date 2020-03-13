import random
import math


class Vocabulary:
    def __init__(self, vocab):
        self.word_to_id = vocab["word_dict"]
        self.id_to_word = dict([(idx, word) for word, idx in self.word_to_id.items()])
        self.char_to_id = vocab["char_dict"]
        self.id_to_char = dict([(idx, word) for word, idx in self.char_to_id.items()])
        self.label_to_id = vocab["label_dict"]
        self.id_to_label = dict([(idx, word) for word, idx in self.label_to_id.items()])
        self.wordvec = vocab["vectors"]
        self.word_weight = vocab["word_weight"]
        self.char_weight = vocab["char_weight"]

    @property
    def word_size(self):
        return len(self.word_to_id)

    @property
    def char_size(self):
        return len(self.char_to_id)

    @property
    def label_size(self):
        return len(self.label_to_id)


class TransVocabulary:
    def __init__(self, vocab):
        self.src_word_to_id = vocab["src_word_dict"]
        self.src_id_to_word = dict([(idx, word) for word, idx in self.src_word_to_id.items()])
        self.src_wordvec = vocab["src_vectors"]
        self.src_word_weight = vocab["src_word_weight"]

        self.tgt_word_to_id = vocab["tgt_word_dict"]
        self.tgt_id_to_word = dict([(idx, word) for word, idx in self.tgt_word_to_id.items()])
        self.tgt_wordvec = vocab["tgt_vectors"]
        self.tgt_word_weight = vocab["tgt_word_weight"]

        self.char_to_id = vocab["char_dict"]
        self.id_to_char = dict([(idx, word) for word, idx in self.char_to_id.items()])
        self.char_weight = vocab["char_weight"]

        self.src_label_to_id = vocab["src_label_dict"]
        self.src_id_to_label = dict([(idx, word) for word, idx in self.src_label_to_id.items()])
        self.tgt_label_to_id = vocab["tgt_label_dict"]
        self.tgt_id_to_label = dict([(idx, word) for word, idx in self.tgt_label_to_id.items()])

    @property
    def src_word_size(self):
        return len(self.src_word_to_id)

    @property
    def tgt_word_size(self):
        return len(self.tgt_word_to_id)

    @property
    def char_size(self):
        return len(self.char_to_id)

    @property
    def src_label_size(self):
        return len(self.src_label_to_id)

    @property
    def tgt_label_size(self):
        return len(self.tgt_label_to_id)


class Dataset:
    def __init__(self, datasets, batch_size=20, train_rate=1.0, shuffle=True):
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._train_set = self._split_train_set(datasets["train"], train_rate)
        self._dev_set = datasets["dev"]
        self._test_set = datasets["test"]

        self._train_batches = self._sample_batches(self._train_set, shuffle=self._shuffle)
        self._dev_batches = self._sample_batches(self._dev_set) if self.dev_size > 0 else None
        self._test_batches = self._sample_batches(self._test_set)

    @staticmethod
    def _split_train_set(dataset, train_rate):
        if train_rate < 1.0:
            train_size = int(len(dataset) * train_rate)
            return dataset[: train_size]
        elif train_rate > 1.0:
            return dataset[:int(train_rate)]
        else:
            return dataset

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return math.ceil(float(len(self._train_set)) / float(self._batch_size))

    @property
    def train_size(self):
        return len(self._train_set)

    @property
    def dev_size(self):
        return len(self._dev_set)

    @property
    def test_size(self):
        return len(self._test_set)

    @property
    def train_set(self):
        return self._train_set

    @property
    def dev_set(self):
        return self._dev_set

    @property
    def test_set(self):
        return self._test_set

    def train_batches(self):
        if self._train_batches is None or len(self._train_batches) < self.num_batches:
            self._train_batches = self._sample_batches(self._train_set, shuffle=self._shuffle)
        return self._train_batches

    def dev_batches(self):
        return self._dev_batches

    def test_batches(self):
        return self._test_batches

    def next_batch(self):
        if self._train_batches is None or len(self._train_batches) == 0:
            self._train_batches = self._sample_batches(self._train_set, shuffle=self._shuffle)
        batch = self._train_batches.pop(0)
        return batch

    def _sample_batches(self, dataset, shuffle=False):
        if shuffle:
            random.shuffle(dataset)
        data_batches = []
        dataset_size = len(dataset)
        for i in range(0, dataset_size, self._batch_size):
            batch_data = dataset[i: i + self._batch_size]
            batch_data = self._process_batch(batch_data)
            data_batches.append(batch_data)
        return data_batches

    @staticmethod
    def _process_batch(batch_data):
        batch_words, batch_words_str, batch_chars, batch_labels = [], [], [], []
        for data in batch_data:
            batch_words.append(data["words"])
            batch_words_str.append(data["words_str"])
            batch_chars.append(data["chars"])
            if "labels" in data:
                batch_labels.append(data["labels"])
        b_words, b_words_len = pad_sequences(batch_words)
        b_words_str, _ = pad_sequences(batch_words_str, pad_tok="")
        b_chars, b_chars_len = pad_char_sequences(batch_chars)
        if len(batch_labels) == 0:
            return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                    "batch_size": len(b_words), "words_str": b_words_str}
        else:
            b_labels, _ = pad_sequences(batch_labels)
            return {"words": b_words, "chars": b_chars, "labels": b_labels, "seq_len": b_words_len,
                    "char_seq_len": b_chars_len, "batch_size": len(b_words), "words_str": b_words_str}


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def boolean_string(bool_str):
    bool_str = bool_str.lower()
    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")
    return bool_str == "true"


def align_data(data):
    """Given dict with lists, creates aligned strings
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                             data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()]) for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()
    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + ' ' * (spacing - len(token) + 1)
        data_aligned[key] = str_aligned
    return data_aligned
