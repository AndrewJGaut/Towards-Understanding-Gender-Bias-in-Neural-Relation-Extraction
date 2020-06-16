import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
import json
import argparse
from Utility import *


trainingdata_name, devdata_name, test_file_name, wordvecdata_name = getDataNames(args)

dataset_name = args.dataset
'''
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
'''
dataset_dir = os.path.join('./data', dataset_name)
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

# The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, trainingdata_name),
                                                        os.path.join(dataset_dir, wordvecdata_name),
                                                        os.path.join(dataset_dir, 'rel2id.json'),
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True)
test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, test_file_name),
                                                       os.path.join(dataset_dir, wordvecdata_name),
                                                       os.path.join(dataset_dir, 'rel2id.json'),
                                                       mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                       shuffle=False)

framework = nrekit.framework.re_framework(train_loader, test_loader)


class model(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")

        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

        # Encoder
        if model.encoder == "pcnn":
            x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        elif model.encoder == "cnn":
            x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
        elif model.encoder == "rnn":
            x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
        elif model.encoder == "birnn":
            x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":
            self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label,
                                                                                   self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label,
                                                                                 self.rel_tot, False, keep_prob=1.0)
        elif model.selector == "ave":
            self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot,
                                                                                 keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot,
                                                                               keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        elif model.selector == "max":
            self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label,
                                                                                 self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label,
                                                                               self.rel_tot, False, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        else:
            raise NotImplementedError

        # Classifier
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot,
                                                                     weights_table=self.get_weights())

    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.ones((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table


model.encoder = args.encoder
model.selector = args.selector

_, _, metrics_dict, new_outputs = framework.test(model,
                                  ckpt="./checkpoint/" + getModelName(args, dataset_name, model.encoder, model.selector),
                                  return_result=True)
#auc, pred_result = framework.test(config, return_result=True)
filename = ""
if args.male_test_files:
    if args.bootstrapped:
        pred_file_name = './test_result/' + getModelName(args, dataset_name, model.encoder, model.selector) + "_male_pred_" + str(args.bootstrap_num) + ".json"
    else:
        pred_file_name = './test_result/' + getModelName(args, dataset_name, model.encoder, model.selector) + "_male_pred" + ".json"
else:
    if args.bootstrapped:
        pred_file_name = './test_result/' + getModelName(args, dataset_name, model.encoder, model.selector) + "_FEmale_pred_" + str(args.bootstrap_num) + ".json"
    else:
        pred_file_name = './test_result/' + getModelName(args, dataset_name, model.encoder, model.selector) + "_FEmale_pred.json"

with open(pred_file_name, 'w') as outfile:
    json.dump(metrics_dict, outfile)
new_output_file_name = pred_file_name.split('.')[0] + '_NEW_OUTPUTS' + '.json'
with open(new_output_file_name, 'w') as outfile:
    json.dump(new_outputs, new_output_file_name)

if args.bootstrapped:
    os.chdir('../../')
