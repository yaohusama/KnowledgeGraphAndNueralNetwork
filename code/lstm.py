import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
num=pickle.load(open("Source/count_all.pkl","rb"))
f = open("Source/data.pkl","rb")
df = pickle.load(f)
data = df["data"]
print("data.shape:",np.array(data).shape)
label=df['label']
x=list(sorted(set(label)))
# print(x)
label=list(map(lambda xx:x.index(xx),label))
num_class=len(list(set(label)))
rnn_unit =200  # hidden layer units
input_size = 400
num_step=128
output_size = num_step
lr = 0.0006  # 学习率
# 训练集划分
def get_train_data(batch_size=num_step, time_step=num_step, train_begin=0, train_end=int(num*0.7)):
    batch_index = []
    data_train = data[train_begin:train_end]
    print("data_train.shape:",np.array(data_train).shape)
    label_train=label[train_begin:train_end]
    # normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    normalized_train_data=data_train
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        # x = normalized_train_data[i:i + time_step, :7]

        x=data_train[i:i+time_step]
        # y = normalized_train_data[i:i + time_step, 7, np.newaxis]
        y=label_train[i:i+time_step]
        train_x.append(x)
        train_y.append(y)
    batch_index.append((len(normalized_train_data) - time_step))
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    return batch_index, train_x, train_y


# 测试集化分
def get_test_data(time_step=num_step, test_begin=int(num*0.7)):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    # normalized_test_data = (data_test - mean) / std  # 标准化
    normalized_test_data=data_test
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step]
        y = normalized_test_data[i * time_step:(i + 1) * time_step]
        test_x.append(x)
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:]))
    test_y.extend((normalized_test_data[(i + 1) * time_step:]))
    return mean, std, test_x, test_y
# 定义变量
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, num_class]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit,])),
    'out': tf.Variable(tf.constant(0.1, shape=[num_class, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    logits = tf.matmul(output, w_out) + b_out
    pred = tf.nn.softmax(logits)

    return pred, final_states


#训练
def train_lstm(batch_size=num_step, time_step=num_step, train_begin=0, train_end=int(num)):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.int64, shape=[None,])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    print(np.array(train_x).shape)
    print(batch_index)
    pred, _ = lstm(X)
    # 计算损失
    # loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    # train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    print("pred.shape:",pred.shape)
    pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                        labels=tf.one_hot(Y,depth=num_class))
    pred_loss = tf.reduce_mean(pred_loss)
    lr = tf.Variable(0.001, dtype=tf.float32)
    regular_train_op = tf.train.AdamOptimizer(lr).minimize(pred_loss)
    correct_label_pred = tf.equal(Y, tf.argmax(pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_file = tf.train.latest_checkpoint('model/')
        if ckpt_file is not None:
            print("loaded")
            saver.restore(sess, ckpt_file)
        # 训练200代
        for i in range(10000):
            # 每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index) - 1):

                _, loss_,acc = sess.run([regular_train_op, pred_loss,label_acc], feed_dict={X: [train_x[batch_index[step]]],
                                                                 Y: train_y[batch_index[step]]})
            print(i, loss_,"acc:",acc)
            if i % 10 == 0:
                print("保存模型：", saver.save(sess, 'model/', global_step=i))

train_lstm()
def test_lstm(batch_size=num_step, time_step=num_step, train_begin=(int)(num*0.7), train_end=num):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.int64, shape=[None,])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    print(np.array(train_x).shape)
    print(batch_index)
    pred, _ = lstm(X)
    # 计算损失
    # loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    # train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    print("pred.shape:",pred.shape)
    pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                        labels=tf.one_hot(Y,depth=num_class))
    pred_loss = tf.reduce_mean(pred_loss)
    # lr = tf.Variable(0.001, dtype=tf.float32)
    # regular_train_op = tf.train.AdamOptimizer(lr).minimize(pred_loss)
    correct_label_pred = tf.equal(Y, tf.argmax(pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_file = tf.train.latest_checkpoint('model/')
        if ckpt_file is not None:
            print("loaded")
            saver.restore(sess, ckpt_file)
            # 每次进行训练的时候，每个batch训练batch_size个样本
            loss=[]
            acc1=[]
            for step in range(len(batch_index) - 1):

                loss_,acc = sess.run([ pred_loss,label_acc], feed_dict={X: [train_x[batch_index[step]]],
                                                                 Y: train_y[batch_index[step]]})
                loss.append(loss_)
                acc1.append(acc)
            print( np.mean(loss),"acc:",np.mean(acc1))

#
test_lstm()

# 预测
def prediction(time_step=num_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        # test_y = np.array(test_y) * std[7] + mean[7]
        # test_predict = np.array(test_predict) * std[7] + mean[7]
        test_y=np.array(test_y)
        test_predict=np.array(test_predict)

        # pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred,
        #                                                     labels=tf.one_hot(test_y, depth=num_class))
        # pred_loss = tf.reduce_mean(pred_loss)
        # lr = tf.Variable(0.001, dtype=tf.float32)
        # regular_train_op = tf.train.AdamOptimizer(lr).minimize(pred_loss)
        correct_label_pred = np.equal(test_y, np.argmax(pred, 1))
        label_acc = np.reduce_mean(np.cast(correct_label_pred, np.float32))
        # acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  #  偏差
        # 以折线图表示结果
        # plt.figure()
        # plt.plot(list(range(len(test_predict))), test_predict, color='b')
        # plt.plot(list(range(len(test_y))), test_y, color='r')
        # plt.show()
        print(label_acc)


# prediction()