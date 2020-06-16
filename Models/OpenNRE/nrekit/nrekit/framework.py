import json
import pdb

import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys
import time





def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.

    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def calculate_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)
def calculate_auc(ground_truth, predicted_scores, positive_label):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ground_truth, predicted_scores, pos_label=positive_label)
    return sklearn.metrics.auc(precision, recall)

class re_model:
    """Basic model class, which contains data input and tensorflow graphs, should be inherited"""

    def __init__(self, train_data_loader, batch_size, max_length=120):
        """
        class construction funciton, model initialization

        Args:
            train_data_loader: a `file_data_loader object`, which could be `npy_data_loader`
                               or `json_file_data_loader`
            batch_size: how many scopes/instances are included in one batch
            max_length: max sentence length, divide sentences into the same length (working
                        part should be finished in `data_loader`)

        Returns:
            None
        """
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.train_data_loader = train_data_loader
        self.rel_tot = train_data_loader.rel_tot
        self.word_vec_mat = train_data_loader.word_vec_mat

    def loss(self):
        """training loss, should be overrided in the subclasses"""
        raise NotImplementedError

    def train_logit(self):
        """training logit, should be overrided in the subclasses"""
        raise NotImplementedError

    def test_logit(self):
        """test logit, should be overrided in the subclasses"""
        raise NotImplementedError


class re_framework:
    """the basic training framework, does all the training and test staffs"""
    MODE_BAG = 0  # Train and test the model at bag level.
    MODE_INS = 1  # Train and test the model at instance level


    def eval_per_relation(self, pred_result):
        """
        NOTE: THIS SHOULD ONLY BE CALLED AT TESTING TIME!!!
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[ [], [], []], 'rec': narray[ [], [], ... ], 'f1': [...], 'auc': [...]}
                Where prec[i] is the precision values at each threshhold for a given relation
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """

        # form the inputs needed for our metrics function

        def getTotalPerRelation():
            '''
            :param rel_2_id:
            :return: the total njumber of instances of this relation in the test data
            '''
            ret_list = [0 for x in range(self.test_data_loader.rel_tot)]
            for k in self.test_data_loader.relfact2scope:
                rel = k
                ret_list[self.test_loader.rel2id[rel]] += 1

            return ret_list

        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec_per_relation = [list() for x in range(self.test_data_loader.rel_tot)]  # make the list per relation
        rec_per_relation = [list() for x in range(self.test_data_loader.rel_tot)]  # make the list per relation
        correct_per_relation = [0 for x in range(self.test_data_loader.rel_tot)]  # make correct per relation
        total_per_relation = getTotalPerRelation()  # an array giving the total relation facts per relatoin (i.e., the total positive examples for each relation)
        # this is a running count of the number of predictions seen for each relation
        # we keep this to be able to calculate precision per-relation at each threshhold
        total_seen_so_far = [0 for x in range(self.test_data_loader.rel_tot)]

        # for aggregate results
        prec = []
        rec = []
        correct = 0
        total = self.test_data_loader.relfact_tot

        for i, item in enumerate(sorted_pred_result):
            rel = item['relation']
            if rel == 0:
                # we don't want to consider the NA relation
                continue
            relation_index = self.test_loader.rel2id[rel]

            #if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                # increment the number correct for this specific relation
            correct_per_relation[relation_index] += item['flag']

            total_seen_so_far[relation_index] += 1

            for rel in range(self.test_data_loader.rel_tot):
                try:

                    # precision is TP (the number 'correctly' classfied above the threshhold) over
                    # TP+FP (which is just equal to the number of times that relation has been predicted positive,
                    # which in our case is the number of times it appears above the threshhold)
                    prec_per_relation[rel].append(float(correct_per_relation[rel]) / float(total_seen_so_far[rel]))
                except:
                    prec_per_relation[rel].append(0)

                try:
                    # recall is TP / (TP + FN) = TP / totalpostivies. In our case, total positives is just the total
                    # numbero f times the relation appears in the 'facts' data structure
                    rec_per_relation[rel].append(float(correct_per_relation[rel]) / float(total_per_relation[rel]))
                except:
                    rec_per_relation[rel].append(0)

            #if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
            correct += item['flag']
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))

        # get the per-relation auc
        auc_per_realtion = [0 for x in range(self.test_data_loader.rel_tot)]
        for i in range(self.test_data_loader.rel_tot):
            try:
                auc_per_realtion[i] = sklearn.metrics.auc(x=rec_per_relation[i], y=prec_per_relation[i])
            except Exception as e:
                # if we get error, just leave the value at 0
                print(e)
                continue

        np_prec_per_relation = np.array(prec_per_relation)
        np_rec_per_relation = np.array(rec_per_relation)
        for i in range(self.test_data_loader.rel_tot):
            np_prec_per_relation[i] = np.array(np_prec_per_relation[i])
            np_rec_per_relation[i] = np.array(np_rec_per_relation[i])

        max_f1_per_relation = [0 for x in range(self.test_data_loader.rel_tot)]
        max_rec_per_relation = [0 for x in range(self.test_data_loader.rel_tot)]
        max_prec_per_relation = [0 for x in range(self.test_data_loader.rel_tot)]
        for i in range(self.test_data_loader.rel_tot):
            try:
                # note that we set this to the max F1 (at some threshhold) for each relation
                max_ind = np.argmax((2 * np_prec_per_relation[i] * np_rec_per_relation[i] / (
                    np_prec_per_relation[i] + np_rec_per_relation[i] + 1e-20)))
                max_prec_per_relation[i] = np_prec_per_relation[i][max_ind].item()
                max_rec_per_relation[i] = np_rec_per_relation[i][max_ind].item()
                max_f1_per_relation[i] = (2 * max_prec_per_relation[i] * max_rec_per_relation[i] / (
                    max_prec_per_relation[i] + max_rec_per_relation[i] + 1e-20))
            except:
                continue
        total_auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        total_max_ind = np.argmax((2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)))
        total_max_prec = np_prec[total_max_ind].item()
        total_max_rec = np_rec[total_max_ind].item()
        total_f1 = (2 * total_max_prec * total_max_rec / (total_max_prec + total_max_rec + 1e-20))

        ret_dict = dict()
        for rel in self.test_data_loader.rel_tot:
            i = self.test_data_loader.rel2id[rel]
            ret_dict[rel] = {'prec': max_prec_per_relation[i], 'recall': max_rec_per_relation[i],
                             'f1': max_f1_per_relation[i], 'auc': auc_per_realtion[i], 'total': total_per_relation[i]}
        ret_dict['total'] = {'prec': total_max_prec, 'recall': total_max_rec, 'f1': total_f1, 'auc': total_auc, 'total': total}
        return ret_dict


    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        """
        class construction funciton, framework initialization

        Args:
            train_data_loader: a `file_data_loader object`, which could be `npy_data_loader`
                               or `json_file_data_loader`
            test_data_loader: similar as the `train_data_loader`
            max_length: max sentence length, divide sentences into the same length (working
                        part should be finished in `data_loader`)
            batch_size: how many scopes/instances are included in one batch

        Returns:
            None
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None  # default graph session

    def one_step_multi_models(self, sess, models, batch_data_gen, run_array, return_label=True):
        """
        run models and multi running tasks via session

        Args:
            sess: tf.Session() that is going to run
            models: a list. this function support multi-model training
            batch_data_gen: `data_loader` to generate batch data
            run_array: a list, contains all the running models or arrays
            return_label: boolean argument. if it is `True`, then the training label
                          will be returned either

        Returns:
            result: a tuple/list contains the result
        """
        feed_dict = {}
        batch_label = []
        for model in models:
            batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size // len(models))
            feed_dict.update({
                model.word: batch_data['word'],
                model.pos1: batch_data['pos1'],
                model.pos2: batch_data['pos2'],
                model.label: batch_data['rel'],
                model.ins_label: batch_data['ins_rel'],
                model.scope: batch_data['scope'],
                model.length: batch_data['length'],
            })
            if 'mask' in batch_data and hasattr(model, "mask"):  # mask data is used in PCNN models
                feed_dict.update({model.mask: batch_data['mask']})
            batch_label.append(batch_data['rel'])
        result = sess.run(run_array, feed_dict)
        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result

    def one_step(self, sess, model, batch_data, run_array):
        """
        run one model and multi running tasks via session, usually used in test operation

        Args:
            sess: tf.Session() that is going to run
            model: one model, inherited from `re_model`
            batch_data: a dict contains the batch data
            run_array: a list, contains all the running models or arrays

        Returns:
            result: a tuple/list contains the result
        """
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})

        result = sess.run(run_array, feed_dict)
        return result

    def train(self, config):
        """
        training function

        Args:
            model: `re_model` that is going to train
            model_name: a string, to identify models, affecting checkpoint saving
            ckpt_dir: checkpoint saving directory
            summary_dir: for tensorboard use, to save summary files
            test_result_dir: directory to store the final results
            learning_rate: learning rate of optimizer
            max_epoch: how many epochs you want to train
            pretrain_model: a string, containing the checkpoint model path and model name
                            e.g. ./checkpoint/nyt_pcnn_one
            test_epoch: when do you want to test the model. default is `1`, which means
                        test the result after every training epoch
            optimizer: training optimizer, default is `tf.train.GradientDescentOptimizer`
            gpu_nums: how many gpus you want to use when training
            not_best_stop: if there is `not_best_stop` epochs that not excel at the model
                           result, the training will be stopped

        Returns:
            None
        """

        model = config['model']
        model_name = config['model_name']
        max_epoch = config['max_epoch']
        learning_rate = config['learning_rate']  # hyperparam
        self.train_data_loader.set_batch_size(config['batch_size'])  # hyperparam
        self.train_data_loader.set_max_len(config['max_length'])  # hyperparam
        ckpt_dir = './checkpoint'
        summary_dir = './summary_dir'
        test_result_dir = './test_result'
        gpu_nums = 1
        optimizer = tf.train.GradientDescentOptimizer
        test_epoch = 1
        not_best_stop = 10 

        # assert(self.train_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)  # allow cpu computing if there is no gpu available
        self.sess = tf.Session(config=config)
        optimizer = optimizer(learning_rate)

        # Multi GPUs
        tower_grads = []
        tower_models = []
        for gpu_id in range(gpu_nums):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader, self.train_data_loader.batch_size // gpu_nums,
                                      self.train_data_loader.max_length)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss()))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss())
                    tf.add_to_collection("train_logit", cur_model.train_logit())

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        train_logit_collection = tf.get_collection("train_logit")
        train_logit = tf.concat(train_logit_collection, 0)

        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        """supporting check the scalars on tensorboard"""
        _output = tf.cast(tf.argmax(train_logit, -1), tf.int32)  # predicted output
        _tot_acc = tf.reduce_mean(
            tf.cast(tf.equal(_output, tower_models[0].label), tf.float32))  # accuracy including N/A relations
        _not_na_acc = tf.reduce_mean(
            tf.cast(tf.logical_and(tf.equal(_output, tower_models[0].label), tf.not_equal(tower_models[0].label, 0)),
                    tf.float32))  # accuracy not including N/A relations

        tf.summary.scalar('tot_acc', _tot_acc)
        tf.summary.scalar('not_na_acc', _not_na_acc)

        # Saver
        saver = tf.train.Saver(max_to_keep=None)

        self.sess.run(tf.global_variables_initializer())



        # Training
        merged_summary = tf.summary.merge_all()  # merge all scalars and histograms
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0  # Stop training after several epochs without improvement.
        global_cnt = 0  # for record summary steps

        # metrics dictionary
        full_metrics_dict = {}

        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    summa, iter_loss, iter_logit, _train_op, iter_label = self.one_step_multi_models(self.sess,
                                                                                                     tower_models,
                                                                                                     self.train_data_loader,
                                                                                                     [merged_summary,
                                                                                                      loss, train_logit,
                                                                                                      train_op])
                except StopIteration:
                    break
                summary_writer.add_summary(summa, global_cnt)
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (
                    epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

            if (epoch + 1) % test_epoch == 0:
                metric, metrics_dict = self.test(model)
                full_metrics_dict['epoch_' + str(epoch)] = metrics_dict
                if metric > best_metric:
                    best_metric = metric
                    best_prec = self.cur_prec
                    best_recall = self.cur_recall
                    print("Best model, storing...")
                    if not os.path.isdir(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                    print("Finish storing")
                    not_best_count = 0
                else:
                    not_best_count += 1

            if not_best_count >= not_best_stop:
                break

            global_cnt += 1

        # save full metrics dict
        with open(model_name + '_model.json', 'w') as fp:
            json.dump(full_metrics_dict, fp)

        print("######")
        print("Finish training " + model_name)
        print("Best epoch auc = %f" % (best_metric))
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.isdir(test_result_dir):
                os.mkdir(test_result_dir)
            np.save(os.path.join(test_result_dir, model_name + "_x.npy"), best_recall)
            np.save(os.path.join(test_result_dir, model_name + "_y.npy"), best_prec)

    def test(self, model, ckpt=None, return_result=False, mode=MODE_BAG):
        """
        test function, to evaluate model

        Args:
            model: a `re_model` which has not been instantiated
            ckpt: whether there is a pretained checkpoing model
            return_result: if True, the predicted result will be returned, either
            mode: basically it is at the bag level

        Returns:
            auc: if return_result is True, return AUC and predicted labels,
                 else return AUC only
        """
        if mode == re_framework.MODE_BAG:
            return self.__test_bag__(model, ckpt=ckpt, return_result=return_result)
        elif mode == re_framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError

    # "NA": 0,
    # "spouse": 1,
    # "": 2,
    # "birthDate": 3,
    # "birthPlace": 4,
    # "hypernym": 5

    # input: 2 arrays, each containing numbers ranging from 0-5 to denote relation
    # output: precision for each relation
    # unclear which relation id is "no relation"
    def calculate_metrics_all_relations(self, ground_truth, predicted_output, predicted_scores):
        '''

        :param ground_truth: ground truth labels
        :param predicted_output: predictions (some integer value representing each relation)
        :param predicted_scores: scores for each correct prediction
        :return: a dictionary represneting a json object containing all metrics we care about (prec, recall, f1, and auc) for every relation
        '''
        # get ground truth counts for each relation
        actual_birthplace_count, actual_birthdate_count, actual_hypernym_count, actual_spouse_count = 0, 0, 0, 0
        for rel in ground_truth:
            if rel == 3:
                actual_birthplace_count += 1
            elif rel == 2:
                actual_birthdate_count += 1
            elif rel == 4:
                actual_hypernym_count += 1
            elif rel == 1:
                actual_spouse_count += 1

        # get predicted counts for each relation
        predicted_birthplace_count, predicted_birthdate_count, predicted_hypernym_count, predicted_spouse_count = 0, 0, 0, 0
        for rel in predicted_output:
            if rel == 3:
                predicted_birthplace_count += 1
            elif rel == 2:
                predicted_birthdate_count += 1
            elif rel == 4:
                predicted_hypernym_count += 1
            elif rel == 1:
                predicted_spouse_count += 1

        # get true positives for each relation
        birthplace_correct, birthdate_correct, hypernym_correct, spouse_correct = 0, 0, 0, 0
        for actual, predicted in zip(ground_truth, predicted_output):
            if actual != predicted:
                continue

            if actual == predicted == 3:
                birthplace_correct += 1
            if actual == predicted == 2:
                birthdate_correct += 1
            if actual == predicted == 4:
                hypernym_correct += 1
            if actual == predicted == 1:
                spouse_correct += 1

        birthplace_precision = birthplace_correct / predicted_birthplace_count
        birthplace_recall = birthplace_correct / actual_birthplace_count
        birthplace_f1 = calculate_f1_score(birthplace_precision, birthplace_recall)
        #birthplace_auc = calculate_auc(ground_truth, predicted_scores, 3)

        birthdate_precision = birthdate_correct / predicted_birthdate_count
        birthdate_recall = birthdate_correct / actual_birthdate_count
        birthdate_f1 = calculate_f1_score(birthdate_precision, birthdate_recall)
        #birthdate_auc = calculate_auc(ground_truth, predicted_scores, 2)

        hypernym_precision = hypernym_correct / predicted_hypernym_count
        hypernym_recall = hypernym_correct / actual_hypernym_count
        hypernym_f1 = calculate_f1_score(hypernym_precision, hypernym_recall)
        #hypernym_auc = calculate_auc(ground_truth, predicted_scores, 4)

        spouse_precision = spouse_correct / predicted_spouse_count
        spouse_recall = spouse_correct / actual_spouse_count
        spouse_f1 = calculate_f1_score(spouse_precision, spouse_recall)
        #spouse_auc = calculate_auc(ground_truth, predicted_scores, 1)

        metrics_dict = {}
        metrics_dict['birthplace'] = {'num_correct': birthplace_correct,
                                      'num_predicted': predicted_birthplace_count,
                                      'num_actual': actual_birthplace_count,
                                      'precision': birthplace_precision,
                                      'recall': birthplace_recall,
                                      'f1_score': birthplace_f1}
                                      #'auc': birthplace_auc}

        metrics_dict['birthdate'] = {'num_correct': birthdate_correct,
                                      'num_predicted': predicted_birthdate_count,
                                      'num_actual': actual_birthdate_count,
                                      'precision': birthdate_precision,
                                      'recall': birthdate_recall,
                                      'f1_score': birthdate_f1}
                                       #'auc': birthdate_auc}

        metrics_dict['hypernym'] = {'num_correct': hypernym_correct,
                                    'num_predicted': predicted_hypernym_count,
                                    'num_actual': actual_hypernym_count,
                                    'precision': hypernym_precision,
                                    'recall': hypernym_recall,
                                    'f1_score': hypernym_f1}
                                     #'auc': hypernym_auc}

        metrics_dict['spouse'] = {'num_correct': spouse_correct,
                                  'num_predicted': predicted_spouse_count,
                                  'num_actual': actual_spouse_count,
                                  'precision': spouse_precision,
                                  'recall': spouse_recall,
                                  'f1_score': spouse_f1}
                                   #'auc': spouse_auc}

        return metrics_dict



    def __test_bag__(self, model, ckpt=None, return_result=False):
        """
        test function at bag level

        Args:
            model: a `re_model` which has not been instantiated
            ckpt: whether there is a pretained checkpoing model
            return_result: if True, the predicted result will be returned, either

        Returns:
            auc: if return_result is True, return AUC and predicted labels,
                 else return AUC only
        """
        print("Testing...")
        if self.sess == None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, self.test_data_loader.batch_size, self.test_data_loader.max_length)
        if not ckpt is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt)

        tot_spouse, tot_spouse_correct = 0, 0
        tot_birthdate, tot_birthdate_correct = 0, 0
        tot_birthplace, tot_birthplace_correct = 0, 0
        tot_hypernym, tot_hypernym_correct = 0, 0

        tot_correct, tot_not_na_correct = 0, 0
        tot, tot_not_na = 0, 0
        entpair_tot = 0
        test_result = []
        pred_result = []

        ground_truth, predicted_output, predicted_scores = [], [], []
        metrics_dict = {}

        for i, batch_data in enumerate(self.test_data_loader):
            iter_logit = self.one_step(self.sess, model, batch_data, [model.test_logit()])[0]
            iter_output = iter_logit.argmax(-1)
            iter_correct = (iter_output == batch_data['rel']).sum()
            iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()

            # print(batch_data['rel'])
            # print(iter_output)

            ground_truth.extend(batch_data['rel'])
            predicted_output.extend(iter_output)
            for index in range(len(iter_output)):
                prediction = iter_output[index]
                predicted_scores.append(iter_logit[index][prediction])

            iter_spouse_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] == 1).sum()
            tot_spouse_correct += iter_spouse_correct
            tot_spouse += np.logical_and(batch_data['rel'], batch_data['rel'] == 1).sum()

            iter_birthdate_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] == 2).sum()
            tot_birthdate_correct += iter_birthdate_correct
            tot_birthdate += np.logical_and(batch_data['rel'], batch_data['rel'] == 2).sum()

            iter_birthplace_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] == 3).sum()
            tot_birthplace_correct += iter_birthplace_correct
            tot_birthplace += np.logical_and(batch_data['rel'], batch_data['rel'] == 3).sum()

            iter_hypernym_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] == 4).sum()
            tot_hypernym_correct += iter_hypernym_correct
            tot_hypernym += np.logical_and(batch_data['rel'], batch_data['rel'] == 4).sum()

            tot_correct += iter_correct
            tot_not_na_correct += iter_not_na_correct
            tot += batch_data['rel'].shape[0]
            tot_not_na += (batch_data['rel'] != 0).sum()
            if tot_not_na > 0:
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (
                i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_rel'][idx][rel], 'relation': batch_data['rel'][idx]})
                    if batch_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]),
                                            'entpair': batch_data['entpair'][idx].encode('utf-8'), 'relation': rel})
                entpair_tot += 1
        metrics_dict = dict()
        try:
            #pdb.set_trace()
            metrics_dict = self.calculate_metrics_all_relations(ground_truth, predicted_output, predicted_scores)
            for relation in metrics_dict:
                print(relation)
                print(metrics_dict[relation])
                print('---')
        except Exception as e:
            print('NOTE!!!!!: Error with metrics dict: {}'.format(e))

        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        self.cur_prec = prec
        self.cur_recall = recall

        my_outputs = self.eval_per_relation(sorted_test_result)

        spouse_percentage = 0
        birthdate_percentage = 0
        birthplace_percentage = 0
        hypernym_percentage = 0
        if (tot_spouse > 0):
            spouse_percentage = float(float(tot_spouse_correct) / float(tot_spouse))
        if (tot_birthdate > 0):
            birthdate_percentage = float(float(tot_birthdate_correct) / float(tot_birthdate))
        if (tot_birthplace > 0):
            birthplace_percentage = float(float(tot_birthplace_correct) / float(tot_birthplace))
        if (tot_hypernym > 0):
            hypernym_percentage = float(float(tot_hypernym_correct) / float(tot_hypernym))
        print('SPOUSE: numcorrect={}, numtot={}, percentage={}'.format(tot_spouse_correct, tot_spouse,
                                                                       spouse_percentage))
        print('BIRTHDATE: numcorrect={}, numtot={}, percentage={}'.format(tot_birthdate_correct,
                                                                          tot_birthdate,
                                                                          birthdate_percentage))
        print('BIRTHPLACE: numcorrect={}, numtot={}, percentage={}'.format(tot_birthplace_correct,
                                                                           tot_birthplace,
                                                                           birthplace_percentage))
        print('HYPERNYM: numcorrect={}, numtot={}, percentage={}'.format(tot_hypernym_correct,
                                                                         tot_hypernym,
                                                                         hypernym_percentage))
        #print('batch_data: ') + batch_data['rel']

        if not return_result:
            return auc, metrics_dict, my_outputs
        else:
            return auc, pred_result, metrics_dict, my_outputs
