import os
import tensorflow as tf


LOGDIR = './mnist/'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

#卷积
def conv_layer(input, size_in, size_out):

    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#全连接
def fc_layer(input, size_in, size_out):
    
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    act = tf.nn.relu(tf.matmul(input, w) + b)

    return act

#全部的网络模型
def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()
    
    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, shape=[None, 10])

    #因为不知道使用一个卷积层好还是两个卷积层好,所以分情况讨论
    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32)
        conv_out = conv_layer(conv1, 32, 64)  
    else:
        conv1 = conv_layer(x_image, 1, 64)
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #从卷积层到全连接层的过渡
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    #因为不知道使用一个全连接层好还是两个全连接层好,所以分情况讨论
    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10)
    else:
        embedding_input = flattened
        embedding_size = 7*7*64
        logits = fc_layer(flattened, 7*7*64, 10)

    #代价函数
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    #训练步骤
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    #正确率
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   
    embedding = tf.Variable(tf.zeros([1024, embedding_size]))
    assignment = embedding.assign(embedding_input)
    
    sess.run(tf.global_variables_initializer())

    #将要展示的东西保存在一个路径
    #如果找不到这个文件夹,会自动创建一个新的文件夹放这些summary文件
    tenboard_dir = './tensorboard/test1/'
    #随着网络运行,要保存summary到当前路径
    #到底要写成什么文件由hparam决定,也就是说创建文件的时候,文件名由学习率,全连接层,卷积层字符串决定
    writer = tf.summary.FileWriter(tenboard_dir + hparam)
    #将回话sess的默认图graph保存起来
    writer.add_graph(sess.graph)


    #迭代训练
    for i in range(2001):
        #取出100个训练样本
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    print('Finish')

#通过学习率,是否使用两个全连接层和两个卷积层来构造一个字符串
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    #造卷积参数和全连接参数
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

'''
这个mnist_board1.py文件是为了展示一下tensorboard基本的启动方式
所以就大概讲一下tensorboard是怎么搭建起来展示网络训练的
但是由于这是第一个py文件,就不做过多展示

这里有三个for循环,第一个for循环是学习率
因为学习率对结果往往有很大影响
那么我们就可以通过遍历一个lr列表来使用tensorboard看看学习率对结果的影响

第二个for循环是是否使用两层全连接层
第三个for循环是是否使用两层卷积层

这里简单展示一下,就先都规定成True
'''
def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:
    # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)

                #将三个for循环扔进函数中形成一些字符串
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)
                
                # Actually run with the new settings
                #将学习率,全连接层,卷积层的字符串打包好传进网络模型函数中
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)

#主函数
if __name__ == '__main__': 
    main()