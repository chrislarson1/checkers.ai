import json
from collections import OrderedDict, namedtuple
from config import *
from utils import *
from tensorflow.python.framework.graph_util import convert_variables_to_constants


class Model:

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _MODEL_DEF_TEMPLATE = {'batch_size': 1, 'dropout': False, 'init': None}

    def __init__(self, name, load_dir, trainable, device):
        self.name = name
        self.trainable = trainable
        self.load_dir = load_dir
        self.write_dir = os.path.join(POLICY_PATH, 'tmp/')
        self._vars = []
        self._ops = []
        self.scopes = []
        self._device = device

    def _build_graph(self):
        pass

    def _constructor(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            try:
                with tf.device(self._device):
                    self._build_graph()
            except AttributeError:
                self._build_graph()

    def init(self, session, load_dir=None):
        load_dir = load_dir or self.load_dir
        for i, layer_scope in enumerate(self.scopes):
            vars_to_load = [v for v in tf.global_variables() if
                            v.name.startswith(layer_scope)]
            self._vars.extend(vars_to_load)
            session.run(tf.variables_initializer(var_list=vars_to_load))
            if load_dir and vars_to_load:
                saver = tf.train.Saver(vars_to_load)
                saver.restore(session, os.path.join(load_dir, self.name))
                # import pprint
                # pp = pprint.PrettyPrinter(indent=3)
                # info('Successfully loaded from {}:'.format(load_dir))
                # pp.pprint(vars_to_load)

    def save_params(self, session, step=None):
        assert self.write_dir
        info('Saving {0} to {1}'.format(self.name, self.write_dir))
        saver = tf.train.Saver(list(set(self._vars)))
        saver.save(session, os.path.join(self.write_dir, self.name), global_step=step)

    def save_graph(self, session:tf.Session, fname:str, var_names:list):
        frozen_graph = convert_variables_to_constants(session, session.graph_def, var_names)
        tf.train.write_graph(frozen_graph, self.write_dir, fname + '.pb', as_text=False)
        tf.train.write_graph(frozen_graph, self.write_dir, fname + '.txt', as_text=True)


class Policy(Model):

    def __init__(self,
                 session:tf.Session,
                 name=None,
                 load_dir=None,
                 trainable=False,
                 selection='greedy',
                 device='GPU'):
        super().__init__(name="POLICY_{}".format(name),
                         load_dir=load_dir,
                         trainable=trainable,
                         device=device)
        self.session = session
        self.state = tf.placeholder(dtype=tf.int32,
                                    shape=(None, 8, 4),
                                    name="state")
        self.action_label = tf.placeholder(dtype=tf.int32,
                                           shape=(None, 128),
                                           name="action")
        self.selection = selection
        self._constructor()

    def _build_graph(self):

        #################### Graph inputs ####################
        self.batch_size = tf.placeholder(shape=(), dtype=tf.float32, name="batch_size")
        self.keep_prob = tf.placeholder(shape=(),
                                        dtype=tf.float32,
                                        name="keep_prob") if self.trainable \
                    else tf.constant(value=1,
                                     dtype=tf.float32,
                                     name="keep_prob")
        self.lr = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")
        self.adv = tf.placeholder(shape=(None), dtype=tf.float32, name="advantage")

        ##################### Data layer #####################
        X = tf.expand_dims(tf.cast(self.state, tf.float32), axis=3)

        ###################### Inception #####################
        with tf.variable_scope("INCEPTION", reuse=False) as scope:
            self.scopes.append(scope.name)
            for i, (ksizes, nkernels) in enumerate(zip(KERNEL_SIZES, N_KERNELS)):
                conv = []
                for ks, nk in zip(ksizes, nkernels):
                    w = tf.get_variable(shape=[ks[0], ks[1], X.shape[-1], nk],
                                        initializer=PARAM_INIT,
                                        trainable=self.trainable,
                                        name='incep_{0}_w_K{1}{2}'.format(i + 1, ks[0], ks[1]))
                    b = tf.get_variable(shape=[nk],
                                        initializer=PARAM_INIT,
                                        trainable=self.trainable,
                                        name='incep_{0}_b_K{1}{2}'.format(i + 1, ks[0], ks[1]))
                    c = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
                    z = INCEP_ACT(c + b)
                    conv.append(z)
                X = tf.concat(conv, axis=3)
            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        ####################### Flatten ######################
        conv_out = tf.contrib.layers.flatten(inputs=X)
        X = conv_out
        hwy_size = X.shape[-1]

        ####################### Highway ######################
        with tf.variable_scope("HIGHWAY", reuse=False) as scope:
            self.scopes.append(scope.name)
            for i in range(HWY_LAYERS):
                with tf.variable_scope('HWY_{}'.format(i)):
                    wh = tf.get_variable(shape=[hwy_size, hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="hwy_w_{}".format(i + 1))
                    bh = tf.get_variable(shape=[hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="hwy_b_{}".format(i + 1))
                    wt = tf.get_variable(shape=[hwy_size, hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="T_w_{}".format(i + 1))
                    T = tf.sigmoid(tf.matmul(X, wt) + HWY_BIAS)
                    H = tf.nn.relu(tf.matmul(X, wh) + bh)
                    X = T * H + (1.0 - T) * X
            X = tf.nn.dropout(X, keep_prob=self.keep_prob)
            X = tf.concat([X, conv_out], axis=1)

        ####################### Output #######################
        with tf.variable_scope("OUTPUT", reuse=False) as scope:
            self.scopes.append(scope.name)
            w = tf.get_variable(shape=[X.shape[-1], 128],
                                initializer=PARAM_INIT,
                                trainable=self.trainable,
                                dtype=tf.float32,
                                name="w_logit")
            b = tf.get_variable(shape=[128],
                                initializer=PARAM_INIT,
                                trainable=self.trainable,
                                dtype=tf.float32,
                                name="b_logit")
            self.logits = tf.add(tf.matmul(X, w), b, name="policy_logits")
            self.softmax = tf.nn.softmax(logits=self.logits, axis=1, name="policy_softmax")
            self.action = tf.argmax(input=self.softmax, axis=1, name="action")
            self.probs, self.actions = tf.nn.top_k(input=self.softmax, k=128, sorted=True)

        ####################### Metrics ######################
        with tf.variable_scope("METRICS", reuse=False) as scope:
            self.scopes.append(scope.name)
            self.top_1_acc = tf.metrics.accuracy(labels=tf.argmax(self.action_label, axis=1),
                                                 predictions=self.action,
                                                 name="accuracy")
            self.top_2_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 2), tf.float32))
            self.top_3_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 3), tf.float32))
            self.top_5_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 5), tf.float32))
            self.top_10_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 10), tf.float32))

        ###################### Optimizer #####################
        if self.trainable:
            with tf.variable_scope("LOSS", reuse=False) as scope:
                self.scopes.append(scope.name)
                self.step = tf.Variable(0, trainable=False)
                self.reg_loss = LAMBDA * tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.global_variables() if v.name.__contains__("w_")])
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.action_label,
                        logits=self.logits,
                        name="cross_entropy")
                self.loss1 = tf.add(self.reg_loss, self.cross_entropy, name="loss1")
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                         name="optimizer_pretrain")
                self.grad_update1 = self.optimizer1.minimize(
                    loss=self.loss1,
                    var_list=tf.trainable_variables(),
                    global_step=self.step,
                    name="grad_update")
                self.gradlogprob_adv = self.adv * tf.log(self.softmax)
                self.pg_loss = tf.reduce_mean(input_tensor=-self.gradlogprob_adv,
                                              axis=1,
                                              name="pg_loss")
                self.optimizer2 = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                            decay=0.99,
                                                            epsilon=1e-5)
                self.policy_update = self.optimizer2.apply_gradients(
                    grads_and_vars=[self.pg_loss, self.vars],
                    global_step=self.step)

    @property
    def vars(self):
        return [v for v in tf.trainable_variables() if
                v.name.lower().__contains__(self.name.lower())]



class Value(Model):

    def __init__(self,
                 session:tf.Session,
                 name=None,
                 load_dir=None,
                 trainable=False,
                 selection='greedy',
                 device='GPU'):
        super().__init__(name="VALUE_{}".format(name), load_dir=load_dir, trainable=trainable, device=device)
        self.session = session
        self.state = tf.placeholder(dtype=tf.int32,
                                    shape=(None, 8, 4),
                                    name="state")
        self.action_label = tf.placeholder(dtype=tf.int32,
                                           shape=(None, 128),
                                           name="action")
        self.selection = selection
        self._constructor()

    def _build_graph(self):

        #################### Graph inputs ####################
        self.batch_size = tf.placeholder(shape=(), dtype=tf.float32, name="batch_size")
        self.keep_prob = tf.placeholder(shape=(),
                                        dtype=tf.float32,
                                        name="keep_prob") if self.trainable \
                    else tf.constant(value=1,
                                     dtype=tf.float32,
                                     name="keep_prob")
        self.lr = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")
        self.adv = tf.placeholder(shape=(None), dtype=tf.float32, name="advantage")

        ##################### Data layer #####################
        X = tf.expand_dims(tf.cast(self.state, tf.float32), axis=3)

        ###################### Inception #####################
        with tf.variable_scope("INCEPTION", reuse=False) as scope:
            self.scopes.append(scope.name)
            for i, (ksizes, nkernels) in enumerate(zip(KERNEL_SIZES, N_KERNELS)):
                conv = []
                for ks, nk in zip(ksizes, nkernels):
                    w = tf.get_variable(shape=[ks[0], ks[1], X.shape[-1], nk],
                                        initializer=PARAM_INIT,
                                        trainable=self.trainable,
                                        name='incep_{0}_w_K{1}{2}'.format(i + 1, ks[0], ks[1]))
                    b = tf.get_variable(shape=[nk],
                                        initializer=PARAM_INIT,
                                        trainable=self.trainable,
                                        name='incep_{0}_b_K{1}{2}'.format(i + 1, ks[0], ks[1]))
                    c = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
                    z = INCEP_ACT(c + b)
                    conv.append(z)
                X = tf.concat(conv, axis=3)
            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        ####################### Flatten ######################
        conv_out = tf.contrib.layers.flatten(inputs=X)
        X = conv_out
        hwy_size = X.shape[-1]

        ####################### Highway ######################
        with tf.variable_scope("HIGHWAY", reuse=False) as scope:
            self.scopes.append(scope.name)
            for i in range(HWY_LAYERS):
                with tf.variable_scope('HWY_{}'.format(i)):
                    wh = tf.get_variable(shape=[hwy_size, hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="hwy_w_{}".format(i + 1))
                    bh = tf.get_variable(shape=[hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="hwy_b_{}".format(i + 1))
                    wt = tf.get_variable(shape=[hwy_size, hwy_size],
                                         initializer=PARAM_INIT,
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         name="T_w_{}".format(i + 1))
                    T = tf.sigmoid(tf.matmul(X, wt) + HWY_BIAS)
                    H = tf.nn.relu(tf.matmul(X, wh) + bh)
                    X = T * H + (1.0 - T) * X
            X = tf.nn.dropout(X, keep_prob=self.keep_prob)
            X = tf.concat([X, conv_out], axis=1)

        ####################### Output #######################
        with tf.variable_scope("OUTPUT", reuse=False) as scope:
            self.scopes.append(scope.name)
            w = tf.get_variable(shape=[X.shape[-1], 128],
                                initializer=PARAM_INIT,
                                trainable=self.trainable,
                                dtype=tf.float32,
                                name="w_logit")
            b = tf.get_variable(shape=[128],
                                initializer=PARAM_INIT,
                                trainable=self.trainable,
                                dtype=tf.float32,
                                name="b_logit")
            self.logits = tf.add(tf.matmul(X, w), b, name="policy_logits")
            self.softmax = tf.nn.softmax(logits=self.logits, axis=1, name="policy_softmax")
            self.action = tf.argmax(input=self.softmax, axis=1, name="action")
            self.probs, self.actions = tf.nn.top_k(input=self.softmax, k=128, sorted=True)

        ####################### Metrics ######################
        with tf.variable_scope("METRICS", reuse=False) as scope:
            self.scopes.append(scope.name)
            self.top_1_acc = tf.metrics.accuracy(labels=tf.argmax(self.action_label, axis=1),
                                                 predictions=self.action,
                                                 name="accuracy")
            self.top_2_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 2), tf.float32))
            self.top_3_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 3), tf.float32))
            self.top_5_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 5), tf.float32))
            self.top_10_acc = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.softmax, tf.argmax(self.action_label, axis=1), 10), tf.float32))

        ###################### Optimizer #####################
        if self.trainable:
            with tf.variable_scope("LOSS", reuse=False) as scope:
                self.scopes.append(scope.name)
                self.step = tf.Variable(0, trainable=False)
                self.reg_loss = LAMBDA * tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.global_variables() if v.name.__contains__("w_")])
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.action_label,
                        logits=self.logits,
                        name="cross_entropy")
                self.loss1 = tf.add(self.reg_loss, self.cross_entropy, name="loss1")
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                         name="optimizer_pretrain")
                self.grad_update1 = self.optimizer1.minimize(
                    loss=self.loss1,
                    var_list=tf.trainable_variables(),
                    global_step=self.step,
                    name="grad_update")
                self.neg_grad_log_prob_adv = self.adv * -tf.log(self.softmax)
                self.SFGE = tf.reduce_mean(input_tensor=self.neg_grad_log_prob_adv,
                                           axis=1,
                                           name="score_func_grad_estimator")



class A2CLoss:

    def __init__(self, policy_network, value_network):
        self.policy = policy_network
        self.value = value_network
        self.__build_graph()

    def __build_graph(self):
        with tf.variable_scope("AC2_LOSS", reuse=False) as scope:
            self.policy.scopes.append(scope.name)
            self.value.scopes.append(scope.name)
            self.lrate = tf.placeholder(shape=(), dtype=tf.float32, name="lrate")
            self.rewards = tf.placeholder(shape=(None), dtype=tf.float32, name="rewards")
            self.baseline = tf.placeholder(shape=(None), dtype=tf.float32, name="value_estimate")
            self.gradlogp = self.adv * tf.log(self.policy.softmax)
            self.logprob = tf.reduce_mean(input_tensor=-self.gradlogprob_adv,
                                          axis=1,
                                          name="pg_loss")
            self.policy_entropy = -tf.reduce_sum(
                self.policy.softmax * tf.log(self.policy.softmax), axis=1
            )

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                       decay=0.99,
                                                       epsilon=1e-5)
            self.policy_update = self.optimizer.apply_gradients(
                grads_and_vars=[self.pg_loss, self.policy.vars],
                global_step=self.policy.step)





if __name__ == '__main__':
    pass
