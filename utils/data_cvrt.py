import os
import re
import glob
import emoji
import codecs
import argparse

emoji_unicode = {v: k for k, v in emoji.EMOJI_UNICODE.items()}


def remove_emoji(line):
    line = "".join(char for char in line if char not in emoji_unicode)
    try:
        pattern = re.compile(u"([\U00002600-\U000027BF])|([\U0001F1E0-\U0001F6FF])")
    except re.error:
        pattern = re.compile(u"([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|"
                             u"([\uD83D][\uDE80-\uDEFF])")
    return pattern.sub(r'', line)


def process_twitter_token(line):
    line = remove_emoji(line)
    line = line.lstrip().rstrip().split("\t")
    if len(line) != 2:
        return None, None
    word, label = line[0], line[1]
    if word.startswith("@") or word.startswith("https://") or word.startswith("http://"):
        return None, None
    if word in ["&gt;", "&quot;", "&lt;", ":D", ";)", ":)", "-_-", "=D", ":'", "-__-", ":P", ":p", "RT", ":-)", ";-)",
                ":(", ":/"]:
        return None, None
    if "&amp;" in word:
        word = word.replace("&amp;", "&")
    if word in ["/", "<"] and label == "O":
        return None, None
    if len(word) == 0:
        return None, None
    return word, label


def convert_wnut_data(file_path, save_path):
    files = ["train.txt", "valid.txt", "test.txt"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in files:
        with codecs.open(os.path.join(save_path, file), mode="w", encoding="utf-8") as f_out:
            with codecs.open(os.path.join(file_path, file), mode="r", encoding="utf-8") as f:
                words, labels = [], []
                for line in f:
                    line = line.lstrip().rstrip()
                    if len(line) == 0:
                        if len(words) == 0:
                            continue
                        str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                        f_out.write("\n".join(str_list) + "\n\n")
                        words, labels = [], []
                    else:
                        word, label = process_twitter_token(line)
                        if word is None or label is None:
                            continue
                        words.append(word)
                        labels.append(label)
                if len(words) != 0:
                    str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                    f_out.write("\n".join(str_list) + "\n\n")


def convert_wsj_data(file_path, save_path):
    # 0-18 for training, 19-21 for development, 22-24 for testing
    train_folders = ["0" + str(i) if i < 10 else str(i) for i in range(19)]
    dev_folders = [str(i) for i in range(19, 22)]
    test_folders = [str(i) for i in range(22, 25)]
    folders_list = [train_folders, dev_folders, test_folders]
    save_files = ["train.txt", "valid.txt", "test.txt"]
    for folders, save_file in zip(folders_list, save_files):
        print(folders, save_file)
        with codecs.open(os.path.join(save_path, save_file), mode="w", encoding="utf-8") as f_out:
            for folder in folders:
                files = glob.glob(os.path.join(file_path, folder) + "/*.pos")
                files.sort()
                for file in files:
                    with codecs.open(file, mode="r", encoding="utf-8") as f:
                        words, labels = [], []
                        for line in f:
                            line = line.lstrip().rstrip()
                            if line.startswith("===========") or len(line) == 0:
                                if len(words) == 0:
                                    continue
                                str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                                f_out.write("\n".join(str_list) + "\n\n")
                                words, labels = [], []
                            else:
                                tokens = line.split(" ")
                                for token in tokens:
                                    token = token.lstrip().rstrip()
                                    if token in ["[", "]"] or len(token) == 0:
                                        continue
                                    idx = token.rfind("/")
                                    word = token[0:idx]
                                    word = word.replace("\\", "")
                                    label = token[idx + 1:]
                                    if "|" in label:
                                        label = label.split("|")[0]
                                    words.append(word)
                                    labels.append(label)
                        if len(words) != 0:
                            str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                            f_out.write("\n".join(str_list) + "\n\n")


def convert_conll(file_path, save_path):
    files = ["train.txt", "valid.txt", "test.txt"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in files:
        with codecs.open(os.path.join(save_path, file), mode="w", encoding="utf-8") as f_out:
            with codecs.open(os.path.join(file_path, file), mode="r", encoding="utf-8") as f:
                words, labels = [], []
                for line in f:
                    line = line.lstrip().rstrip()
                    if len(line) == 0 or line.startswith("-DOCSTART-"):
                        if len(words) == 0:
                            continue
                        str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                        f_out.write("\n".join(str_list) + "\n\n")
                        words, labels = [], []
                    else:
                        word, *_, label = line.split(" ")
                        if "page=http" in word or "http" in word:
                            continue
                        words.append(word)
                        labels.append(label)
                if len(words) != 0:
                    str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                    f_out.write("\n".join(str_list) + "\n\n")


def convert_ontonotes(file_path, save_path, token="pos"):
    files = ["ontonotes.train.iob", "ontonotes.development.iob", "ontonotes.test.iob"]
    save_files = ["train.txt", "valid.txt", "test.txt"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file, save_file in zip(files, save_files):
        print(file, save_file)
        with codecs.open(os.path.join(save_path, save_file), mode="w", encoding="utf-8") as f_out:
            with codecs.open(os.path.join(file_path, file), mode="r", encoding="utf-8") as f:
                words, labels = [], []
                for line in f:
                    line = line.lstrip().rstrip()
                    if len(line) == 0 or line.startswith("#begin"):
                        if len(words) == 0:
                            continue
                        str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                        f_out.write("\n".join(str_list) + "\n\n")
                        words, labels = [], []
                    else:
                        word, pos, ner = line.split(" ")
                        words.append(word)
                        if token == "pos":
                            labels.append(pos)
                        else:
                            labels.append(ner)
                if len(words) != 0:
                    str_list = ["{}\t{}".format(word, label) for word, label in zip(words, labels)]
                    f_out.write("\n".join(str_list) + "\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='file path')
    parser.add_argument('--save_path', type=str, required=True, help='save path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    args = parser.parse_args()
    if args.dataset == 'wsj':
        convert_wsj_data(args.file_path, args.save_path)
    elif args.dataset == 'conll':
        convert_conll(args.file_path, args.save_path)
    elif args.dataset == 'wnut':
        convert_wnut_data(args.file_path, args.save_path)
    elif args.dataset == 'ontonotes':
        convert_ontonotes(args.file_path, args.save_path)
    else:
        raise ValueError('Unknown dataset...')
