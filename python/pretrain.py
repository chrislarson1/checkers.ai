import tqdm
from utils import *
from config import *
from parser import parse
from model import Policy

def train():
    global lr, epoch, ACC, LOSS, PASS_ANNEAL_RATE, FAIL_ANNEAL_RATE
    info("Building Graph...")
    TF_CONFIG = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    session = tf.Session(config=TF_CONFIG)
    policy = Policy(session=session,
                    load_dir=load_dir,
                    trainable=True,
                    device="GPU:0")
    session.run(tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ))
    policy.init(session=session)
    warn("UNITITIALIZED VARIABLES:")
    warn(str(session.run(tf.report_uninitialized_variables())))
    warn("TRAINABLE VARIABLES:")
    [warn("Variable name: {}, Variable: {}".format(
        v.name, v)) for v in tf.trainable_variables()]

    def run_minibatches(X, y, desc:str, train:bool, bsize:int, shuffle=True):
        assert len(X) == len(y)
        if shuffle:
            idx = list(range(len(X)))
            np.random.shuffle(idx)
            X, y= X[idx], y[idx]
        n_batches = len(X) // bsize
        fetches = [policy.grad_update1, policy.loss1, policy.top_1_acc, policy.top_5_acc] if train \
                    else [policy.loss1, policy.top_1_acc, policy.top_5_acc]
        bar = tqdm.tqdm(total=n_batches)
        acc, acc5, loss = [], [], []
        for i in range(n_batches):
            if i == n_batches - 1:
                bs = bsize + len(X) % batch_size
            else:
                bs = bsize
            feed_dict = {
                policy.state: X[i * bsize: i * bsize + bs],
                policy.action_label: y[i * bsize: i * bsize + bs],
                policy.keep_prob: KEEP_PROB if train else 1,
                policy.lr: lr
            }
            result = session.run(fetches, feed_dict=feed_dict)
            loss.append(result[-3][0])
            acc.append(result[-2][0])
            acc5.append(result[-1])
            bar.set_description("Epoch: %d  | Mode: %s  |  acc: %.5f  |  acc_top_5: %.5f  |  loss: %.5f  |  lr: %.7f" % (
                epoch, desc, np.mean(acc), np.mean(acc5), np.mean(loss), lr))
            bar.update(1)
        bar.close()
        return np.mean(loss, keepdims=False), np.mean(acc, keepdims=False)

    for epoch in range(epochs):
        lossTr, accTr = run_minibatches(xTr, yTr, train=True, bsize=batch_size, shuffle=True, desc='train')
        lossTr, accTr = run_minibatches(xTr, yTr, train=False, bsize=512, shuffle=False, desc='train_eval')
        lossCv, accCv = run_minibatches(xCv, yCv, train=False, bsize=512, shuffle=False, desc='valid_eval')
        if accCv > ACC:
            LOSS = lossCv
            ACC = accCv
            lr *= PASS_ANNEAL_RATE
            policy.save_params(session=session)
            # policy.save_graph(session=session,
            #                   fname='policy',
            #                   var_names=[v.name for v in policy._vars])
        else:
            lr *= FAIL_ANNEAL_RATE
        if lr < MIN_LRATE:
            break

        if epoch == 100:
            PASS_ANNEAL_RATE = 0.95
            FAIL_ANNEAL_RATE = 0.80

    print('\n')
    warn("EVALUATION:")
    policy.init(session=session, load_dir=policy.write_dir)
    lossTr, accTr = run_minibatches(xTr, yTr, train=False, bsize=512, shuffle=False, desc='train')
    lossCv, accCv = run_minibatches(xCv, yCv, train=False, bsize=512, shuffle=False, desc='valid')
    lossTe, accTe = run_minibatches(xTe, yTe, train=False, bsize=512, shuffle=False, desc='test')
    warn("TRAIN stats: Loss: %.5f  |  Acc: %.5f" % (lossTr, accTr))
    warn("VALID stats: Loss: %.5f  |  Acc: %.5f" % (lossCv, accCv))
    warn(" TEST stats: Loss: %.5f  |  Acc: %.5f" % (lossTe, accTe))




if __name__ == '__main__':

    # Training parameters
    epochs = 0
    epoch = 0
    batch_size = 128
    cv_split = (0.85, 0.1, 0.05)
    lr = LRATE
    LOSS = np.inf
    ACC = 0
    ctx = 'GPU:0'
    load_dir = os.path.join(POLICY_PATH, '81.9acc')

    # Load training data
    if not os.path.isfile(DFILE): parse()
    data = np.load(DFILE)
    states, actions = data['states'].reshape(-1, 8, 4).astype(np.int32), \
                      data['actions'].reshape(-1, 128).astype(np.int32)
    del data
    nTr = int(cv_split[0] * len(states))
    nCv = int(cv_split[1] * len(states))
    xTr, yTr = states[:nTr], actions[:nTr]
    xCv, yCv = states[nTr:nTr + nCv], actions[nTr:nTr + nCv]
    xTe, yTe = states[nTr + nCv:], actions[nTr + nCv:]
    info("Train_set: {0}, {1}".format(xTr.shape, yTr.shape))
    info("Valid_set: {0}, {1}".format(xCv.shape, yCv.shape))
    info("Test_set: {0}, {1}".format(xTe.shape, yTe.shape))
    train()
