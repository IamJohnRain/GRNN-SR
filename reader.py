# -*- coding: utf-8 -*-
"""
Created on Feb 21  10:16  2017

@author: chuito
"""

import numpy as np
import re
import os
from tensorflow.contrib import learn

# the directory of pretrained word embedding such as GloVe, Google word2vec
word2vec_file = r"./data/glove.txt"

TOKENIZER_RE = re.compile(r"[a-zA-Z0-9,.!?_]+")


def clean_str(string):
    string = string.replace("_", " ")
    string = string.replace(",", " , ").replace(".", " ").replace("!", " ").replace("?", " ")
    string = string.replace("-", " ").replace("'d", " would").replace("'ll", " will").replace("can't", "can not")
    string = string.replace("n't", " not")
    string = string.replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'m", " am")
    string = string.replace("'", " ").lower()
    # words = [w for w in string.split(" ") if not re.findall("\d+", w)]
    words = re.findall(TOKENIZER_RE, string)
    # words = re.findall("[a-zA-Z,]+", string)
    words = [w for w in words if len(w) > 1 or w in {'i', 'a', ',', '.', '!', '?'}]
    string = " ".join(words)
    return string.strip()


def tokenizer(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)


class BaseReader(object):
    def __init__(self, data_path):
        self.embed_file = os.path.join(data_path, "embed.txt")
        self.word2vec_file = word2vec_file
        self.id2word = []
        self.vocab_size = 0

    def batch_iter(self, batch_size, data_type="train", shuffle=True):
        """
        Generate a batch iterator for a dataset.
        """
        data = np.array(list(zip(*self.fetch_data(data_type))))
        data_size = len(data)
        num_batches_per_epoch = int(np.ceil(data_size / batch_size))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]

    def load_initial_embed(self):
        word2id = {w: i for i, w in enumerate(self.id2word)}
        initial_embed = np.random.uniform(-0.01, 0.01, [self.vocab_size, 300])
        with open(self.embed_file, "r") as f:
            for line in f.readlines():
                word, vs = line.strip().split(" ", 1)
                if word in word2id:
                    initial_embed[word2id[word]] = [float(v) for v in vs.split(" ")]
        initial_embed = np.array(initial_embed, np.float32)
        return initial_embed


class SSTReader(BaseReader):
    def __init__(self, data_path, update_embedding=False, binary=True):
        """
        :param data_path: path of dataset
        :param update_embedding: wheather to update embed_file
        :param binary: True for binary classifcation, False for 5-class classification
        """
        super(SSTReader, self).__init__(data_path)
        # self.negators = {"not", "no", "never", "neither", "hardly", "seldom", "nothing"}
        # self.intensities = {"very", "greatly", "absolutely", "completely", "terribly", "fairly",
        #                     "highly", "more", "really"}

        self.negators = {"no", "not", "none", "never", "neither", "nobody", "nothing", "nowhere", "seldom",
                         "scarcely", "hardly", "barely", "is not", "cannot", "may not", "could not",
                         "would not", "did not", "do not", "does not", "was not", "are not", "were not"}
        self.intensities = {"awfully", "extraordinary", "unusual", "much", "rather", "very", "entirely",
                            "greatly", "really", "exceedingly", "too", "completely", "terribly",
                            "perfectly", "quite", "certainly", "especially", "extremely", "fairly",
                            "highly", "increasingly", "much more", "particularly", "probably", "more",
                            "absolutely", "intensely", "supremely", "most", "pretty"}

        self.load_data(data_path, update_embedding, binary)

    def file2data(self, file_name, binary=True):
        with open(file_name, "r", encoding="utf-8") as f:
            data = [s.strip().split("|||") for s in f.readlines()]

        if binary:
            polarity = np.sign([int(d[1])-2 for d in data if d[1] and int(d[1]) != 2])
            text = [d[0] for d in data if d[1] and int(d[1]) != 2]
        else:
            polarity = [int(d[1]) for d  in data if d[1]]
            text = [d[0] for d in data if d[1]]
        text = [clean_str(s) for s in text]
        return text, polarity

    def filter_data(self, texts, filter_set):
        filter_text_idx = []
        for n, text in enumerate(texts):
            for w in text.split(" "):
                if w in filter_set:
                    filter_text_idx.append(n)
                    break
        return filter_text_idx

    def load_data(self, data_path, update_embedding, binary):
        self.metadata_file = os.path.join(data_path, "metadata.tsv")
        self.nega_metadata_file = os.path.join(data_path, "nega_metadata.tsv")
        self.inten_metadata_file = os.path.join(data_path, "inten_metadata.tsv")
        if binary:
            self.train_file = os.path.join(data_path, "binary-train.txt")
            # self.train_file = os.path.join(data_path, "raw-sent-train.txt")
        else:
            self.train_file = os.path.join(data_path, "5class-train.txt")
            # self.train_file = os.path.join(data_path, "raw-sent-train.txt")
        self.valid_file = os.path.join(data_path, "valid.txt")
        self.test_file = os.path.join(data_path, "test.txt")
        self.vocab_file = os.path.join(data_path, "vocab.tsv")

        train_text, train_pola = self.file2data(self.train_file, binary)
        valid_text, valid_pola = self.file2data(self.valid_file, binary)
        test_text, test_pola = self.file2data(self.test_file, binary)

        max_sent_length = max(map(lambda x: len(x.split(" ")), train_text + valid_text + test_text))
        self.sequence_length = max_sent_length

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sent_length,
                                                                  tokenizer_fn=tokenizer)
        vocab_processor.fit(train_text + valid_text + test_text)
        self.word2id = vocab_processor.vocabulary_._mapping
        self.id2word = vocab_processor.vocabulary_._reverse_mapping
        self.vocab_size = len(self.id2word)

        self.id2pola = sorted(list(set(train_pola)))
        self.pola2id = {p: i for i, p in enumerate(self.id2pola)}
        self.polarity_size = len(self.id2pola)
        print("Polarity:", self.pola2id)

        # write vocab_file
        with open(self.vocab_file, "w") as f:
            for word in self.id2word:
                f.write(word + "\n")

        train = np.array(list(vocab_processor.transform(train_text)))
        valid = np.array(list(vocab_processor.transform(valid_text)))
        test  = np.array(list(vocab_processor.transform(test_text)))

        self.train_x = train
        self.train_l = np.array([len(s.split(" ")) for s in train_text])
        self.train_p = np.array([self.pola2id[i] for i in train_pola])

        self.valid_x = valid
        self.valid_l = np.array([len(s.split(" ")) for s in valid_text])
        self.valid_p = np.array([self.pola2id[i] for i in valid_pola])

        self.test_x = test
        self.test_l = np.array([len(s.split(" ")) for s in test_text])
        self.test_p = np.array([self.pola2id[i] for i in test_pola])

        # negation subset
        nega_train_idx = self.filter_data(train_text, self.negators)
        self.nega_train_x = self.train_x[nega_train_idx]
        self.nega_train_l = self.train_l[nega_train_idx]
        self.nega_train_p = self.train_p[nega_train_idx]

        nega_valid_idx = self.filter_data(valid_text, self.negators)
        self.nega_valid_x = self.valid_x[nega_valid_idx]
        self.nega_valid_l = self.valid_l[nega_valid_idx]
        self.nega_valid_p = self.valid_p[nega_valid_idx]

        nega_test_idx = self.filter_data(test_text, self.negators)
        self.nega_test_x = self.test_x[nega_test_idx]
        self.nega_test_l = self.test_l[nega_test_idx]
        self.nega_test_p = self.test_p[nega_test_idx]

        # intensity subset
        inten_train_idx = self.filter_data(train_text, self.intensities)
        self.inten_train_x = self.train_x[inten_train_idx]
        self.inten_train_l = self.train_l[inten_train_idx]
        self.inten_train_p = self.train_p[inten_train_idx]

        inten_valid_idx = self.filter_data(valid_text, self.intensities)
        self.inten_valid_x = self.valid_x[inten_valid_idx]
        self.inten_valid_l = self.valid_l[inten_valid_idx]
        self.inten_valid_p = self.valid_p[inten_valid_idx]

        inten_test_idx = self.filter_data(test_text, self.intensities)
        self.inten_test_x = self.test_x[inten_test_idx]
        self.inten_test_l = self.test_l[inten_test_idx]
        self.inten_test_p = self.test_p[inten_test_idx]

        if 0 in self.train_l:
            indexes = np.array([l > 0 for l in self.train_l])
            self.train_x = self.train_x[indexes]
            self.train_l = self.train_l[indexes]
            self.train_p = self.train_p[indexes]

        if 0 in self.valid_l:
            indexes = np.array([l > 0 for l in self.valid_l])
            self.valid_x = self.valid_x[indexes]
            self.valid_l = self.valid_l[indexes]
            self.valid_p = self.valid_p[indexes]

        if 0 in self.test_l:
            indexes = np.array([l > 0 for l in self.test_l])
            self.test_x = self.test_x[indexes]
            self.test_l = self.train_l[indexes]
            self.test_p = self.test_p[indexes]

        with open(self.metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.test_p, self.test_x, self.test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Max word counts: {} Vocab: {}".format(max_sent_length, len(self.word2id)))
        print("Total Sentences: {}/{}/{}".format(*map(len, [self.train_x, self.valid_x, self.test_x])))

        with open(self.nega_metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.nega_test_p, self.nega_test_x, self.nega_test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Negation Sentences: {}/{}/{}".format(*map(len, [self.nega_train_x, self.nega_valid_x, self.nega_test_x])))

        with open(self.inten_metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.inten_test_p, self.inten_test_x, self.inten_test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Intensity Sentences: {}/{}/{}".format(*map(len, [self.inten_train_x, self.inten_valid_x, self.inten_test_x])))

        if update_embedding:
            word2vec = {}
            with open(self.word2vec_file, "r") as f:
                for line in f:
                    word, vec_str = line.strip().split(" ", 1)
                    vec = [float(x) for x in vec_str.split(" ")]
                    if word in self.word2id:
                        word2vec[word] = vec
            print(len(word2vec), "words have pre-trained vector.")
            with open(self.embed_file, "w") as f:
                for w, vec in word2vec.items():
                    f.write(w + " " + " ".join(map(str, vec)) + "\n")

    def fetch_data(self, data_type="train"):
        if data_type == "train":
            return self.train_x, self.train_l, self.train_p
        elif data_type == "test":
            return self.test_x, self.test_l, self.test_p
        elif data_type == "valid":
            return self.valid_x, self.valid_l, self.valid_p
        elif data_type == "nega_train":
            return self.nega_train_x, self.nega_train_l, self.nega_train_p
        elif data_type == "nega_valid":
            return self.nega_valid_x, self.nega_valid_l, self.nega_valid_p
        elif data_type == "nega_test":
            return self.nega_test_x, self.nega_test_l, self.nega_test_p
        elif data_type == "inten_train":
            return self.inten_train_x, self.inten_train_l, self.inten_train_p
        elif data_type == "inten_valid":
            return self.inten_valid_x, self.inten_valid_l, self.inten_valid_p
        elif data_type == "inten_test":
            return self.inten_test_x, self.inten_test_l, self.inten_test_p
        else:
            raise Exception("Unsupported datatype: {}".format(data_type))

############### Test ################

# reader = SSTReader(data_path="./data/SST/", update_embedding=False, binary=False)
# batches = reader.batch_iter(32, data_type="test", shuffle=False)
# train_x, train_l, train_p= zip(*next(batches))
# i = 10
# print(train_x[i].shape)
# print(train_x[i])
# print("Length:", train_l[i])
# print("Polarity:", train_p[i], reader.id2pola[train_p[i]])
# print(" ".join([reader.id2word[x] for x in train_x[i][:train_l[i]]]))
# print(" ".join(map(str, [reader.word2score.get(reader.id2word[x], "s") for x in train_x[i][:train_l[i]]])))


class MovieReader(BaseReader):

    def __init__(self, data_path, update_embedding=False):
        """
        :param data_path: path of dataset
        :param update_embedding: wheather to update embed_file
        :param binary: True for binary classifcation, False for 5-class classification
        """
        super(MovieReader, self).__init__(data_path)
        # self.negators = {"not", "no", "never", "neither", "hardly", "seldom", "nothing"}
        # self.intensities = {"very", "greatly", "absolutely", "completely", "terribly", "fairly",
        #                     "highly", "more", "really"}

        self.negators = {"no", "not", "none", "never", "neither", "nobody", "nothing", "nowhere", "seldom",
                         "scarcely", "hardly", "barely", "is not", "cannot", "may not", "could not",
                         "would not", "did not", "do not", "does not", "was not", "are not", "were not"}
        self.intensities = {"awfully", "extraordinary", "unusual", "much", "rather", "very", "entirely",
                            "greatly", "really", "exceedingly", "too", "completely", "terribly",
                            "perfectly", "quite", "certainly", "especially", "extremely", "fairly",
                            "highly", "increasingly", "much more", "particularly", "probably", "more",
                            "absolutely", "intensely", "supremely", "most", "pretty"}

        self.load_data(data_path, update_embedding)

    def file2data(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            data = [s.strip().split("|||") for s in f.readlines()]

        polarity = [int(d[0]) for d in data]
        text = [d[1] for d in data]
        text = [clean_str(s) for s in text]
        return text, polarity

    def filter_data(self, texts, filter_set):
        filter_text_idx = []
        for n, text in enumerate(texts):
            for w in text.split(" "):
                if w in filter_set:
                    filter_text_idx.append(n)
                    break
        return filter_text_idx

    def load_data(self, data_path, update_embedding):
        self.metadata_file = os.path.join(data_path, "metadata.tsv")
        self.nega_metadata_file = os.path.join(data_path, "nega_metadata.tsv")
        self.inten_metadata_file = os.path.join(data_path, "inten_metadata.tsv")

        self.train_file = os.path.join(data_path, "train.txt")
        self.valid_file = os.path.join(data_path, "valid.txt")
        self.test_file = os.path.join(data_path, "test.txt")
        self.vocab_file = os.path.join(data_path, "vocab.tsv")

        train_text, train_pola = self.file2data(self.train_file)
        valid_text, valid_pola = self.file2data(self.valid_file)
        test_text, test_pola = self.file2data(self.test_file)

        max_sent_length = max(map(lambda x: len(x.split(" ")), train_text + valid_text + test_text))
        self.sequence_length = max_sent_length

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sent_length,
                                                                  tokenizer_fn=tokenizer)
        vocab_processor.fit(train_text + valid_text + test_text)
        self.word2id = vocab_processor.vocabulary_._mapping
        self.id2word = vocab_processor.vocabulary_._reverse_mapping
        self.vocab_size = len(self.id2word)

        self.id2pola = sorted(list(set(train_pola)))
        self.pola2id = {p: i for i, p in enumerate(self.id2pola)}
        self.polarity_size = len(self.id2pola)
        print("Polarity:", self.pola2id)

        # write vocab_file
        with open(self.vocab_file, "w") as f:
            for word in self.id2word:
                f.write(word + "\n")

        train = np.array(list(vocab_processor.transform(train_text)))
        valid = np.array(list(vocab_processor.transform(valid_text)))
        test  = np.array(list(vocab_processor.transform(test_text)))

        self.train_x = train
        self.train_l = np.array([len(s.split(" ")) for s in train_text])
        self.train_p = np.array([self.pola2id[i] for i in train_pola])

        self.valid_x = valid
        self.valid_l = np.array([len(s.split(" ")) for s in valid_text])
        self.valid_p = np.array([self.pola2id[i] for i in valid_pola])

        self.test_x = test
        self.test_l = np.array([len(s.split(" ")) for s in test_text])
        self.test_p = np.array([self.pola2id[i] for i in test_pola])

        # negation subset
        nega_train_idx = self.filter_data(train_text, self.negators)
        self.nega_train_x = self.train_x[nega_train_idx]
        self.nega_train_l = self.train_l[nega_train_idx]
        self.nega_train_p = self.train_p[nega_train_idx]

        nega_valid_idx = self.filter_data(valid_text, self.negators)
        self.nega_valid_x = self.valid_x[nega_valid_idx]
        self.nega_valid_l = self.valid_l[nega_valid_idx]
        self.nega_valid_p = self.valid_p[nega_valid_idx]

        nega_test_idx = self.filter_data(test_text, self.negators)
        self.nega_test_x = self.test_x[nega_test_idx]
        self.nega_test_l = self.test_l[nega_test_idx]
        self.nega_test_p = self.test_p[nega_test_idx]

        # intensity subset
        inten_train_idx = self.filter_data(train_text, self.intensities)
        self.inten_train_x = self.train_x[inten_train_idx]
        self.inten_train_l = self.train_l[inten_train_idx]
        self.inten_train_p = self.train_p[inten_train_idx]

        inten_valid_idx = self.filter_data(valid_text, self.intensities)
        self.inten_valid_x = self.valid_x[inten_valid_idx]
        self.inten_valid_l = self.valid_l[inten_valid_idx]
        self.inten_valid_p = self.valid_p[inten_valid_idx]

        inten_test_idx = self.filter_data(test_text, self.intensities)
        self.inten_test_x = self.test_x[inten_test_idx]
        self.inten_test_l = self.test_l[inten_test_idx]
        self.inten_test_p = self.test_p[inten_test_idx]

        if 0 in self.train_l:
            indexes = np.array([l > 0 for l in self.train_l])
            self.train_x = self.train_x[indexes]
            self.train_l = self.train_l[indexes]
            self.train_p = self.train_p[indexes]

        if 0 in self.valid_l:
            indexes = np.array([l > 0 for l in self.valid_l])
            self.valid_x = self.valid_x[indexes]
            self.valid_l = self.valid_l[indexes]
            self.valid_p = self.valid_p[indexes]

        if 0 in self.test_l:
            indexes = np.array([l > 0 for l in self.test_l])
            self.test_x = self.test_x[indexes]
            self.test_l = self.train_l[indexes]
            self.test_p = self.test_p[indexes]

        with open(self.metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.test_p, self.test_x, self.test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Max word counts: {} Vocab: {}".format(max_sent_length, len(self.word2id)))
        print("Total Sentences: {}/{}/{}".format(*map(len, [self.train_x, self.valid_x, self.test_x])))

        with open(self.nega_metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.nega_test_p, self.nega_test_x, self.nega_test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Negation Sentences: {}/{}/{}".format(*map(len, [self.nega_train_x, self.nega_valid_x, self.nega_test_x])))

        with open(self.inten_metadata_file, "w") as f:
            f.write("ID\tPolarity\tText\n")
            for i, (p, sent, l) in enumerate(zip(self.inten_test_p, self.inten_test_x, self.inten_test_l)):
                pola = self.id2pola[p]
                text = " ".join([self.id2word[w] for w in sent[:l]])
                f.write("{}\t{}\t{}\n".format(i, pola, text))

        print("Intensity Sentences: {}/{}/{}".format(*map(len, [self.inten_train_x, self.inten_valid_x, self.inten_test_x])))

        if update_embedding:
            word2vec = {}
            with open(self.word2vec_file, "r", encoding="utf-8") as f:
                for line in f:
                    word, vec_str = line.strip().split(" ", 1)
                    vec = [float(x) for x in vec_str.split(" ")]
                    if word in self.word2id:
                        word2vec[word] = vec
            print(len(word2vec), "words have pre-trained vector.")
            with open(self.embed_file, "w") as f:
                for w, vec in word2vec.items():
                    f.write(w + " " + " ".join(map(str, vec)) + "\n")

    def fetch_data(self, data_type="train"):
        if data_type == "train":
            return self.train_x, self.train_l, self.train_p
        elif data_type == "test":
            return self.test_x, self.test_l, self.test_p
        elif data_type == "valid":
            return self.valid_x, self.valid_l, self.valid_p
        elif data_type == "nega_train":
            return self.nega_train_x, self.nega_train_l, self.nega_train_p
        elif data_type == "nega_valid":
            return self.nega_valid_x, self.nega_valid_l, self.nega_valid_p
        elif data_type == "nega_test":
            return self.nega_test_x, self.nega_test_l, self.nega_test_p
        elif data_type == "inten_train":
            return self.inten_train_x, self.inten_train_l, self.inten_train_p
        elif data_type == "inten_valid":
            return self.inten_valid_x, self.inten_valid_l, self.inten_valid_p
        elif data_type == "inten_test":
            return self.inten_test_x, self.inten_test_l, self.inten_test_p
        else:
            raise Exception("Unsupported datatype: {}".format(data_type))


############### Test ################

# reader = MovieReader(data_path="./data/Movie/", update_embedding=False)
# batches = reader.batch_iter(32, data_type="test", shuffle=False)
# train_x, train_l, train_p= zip(*next(batches))
# i = 10
# print(train_x[i].shape)
# print(train_x[i])
# print("Length:", train_l[i])
# print("Polarity:", train_p[i], reader.id2pola[train_p[i]])
# print(" ".join([reader.id2word[x] for x in train_x[i][:train_l[i]]]))
# print(" ".join(map(str, [reader.word2score.get(reader.id2word[x], "s") for x in train_x[i][:train_l[i]]])))
