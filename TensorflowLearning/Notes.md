在Tensorflow程序中，系统会自动维护一个默认的计算图，通过`tf.get_default_graph`函数可以获取当前默认的计算图。支持通过`tf.Graph`函数来生成新的计算图，不同计算图上的张量和运算都不会共享。

```
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量'v'，并设置初始值为0
    v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer(tf.float32))

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量'v'，并设置初始值为1
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer(tf.float32))

# 在计算图g1中读取v的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))

# 在计算图g2中读取v的值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))
```
返回值

```
[0.]
[1.]
```
在一个计算图中，可以通过集合（collection）来管理不同类别的资源。

| 集合名称 | 集合内容 | 使用场景 |
| --- | --- | --- |
| tf.GraphKeys.VARIABLES | 所有变量 | 持久化Tensorflow模型 |
| tf.GraphKeys.TRAINABLE_VARIABLES | 可学习的变量（一般指神经网络中的参数） | 模型训练、生成模型可视化内容 |
| tf.GraphKeys.SUMMARIES | 日志生成的相关变量 | TensorFlow计算可视化 |
| tf.GraphKeys.QUEUE_RUNNERS  | 处理输入的QueueRunner  | 处理输出 |
| tf.GraphKeys.MOVING_AVERAGE_VARIABLES | 所有计算了滑动平均值的变量 | 计算变蓝的滑动平均值 |

张量的三个属性：
1. 名字：张量的命名可以通过“node:src_output”的形式来给出，其中node为节点的名称，src_output表示当前张量来自节点的第几个输出（编号从0开始）
2. 维度：shape
3. 类型：type。支持14种类型，实数（tf.float32, tf.float64）、整数(tf.int8、tf.int16、tf.int32、tf.int64、tf.uint8)、布尔型（tf.bool）和复数（tf.complex64、tf.complex128）

当构建机器学习模型时，比如神经网络，可以通过变量声明函数中的trainable参数来区分需要优化的参数（比如神经网络中的参数）和其它参数（比如迭代轮数）。如果声明变量时参数trainable为True，那么这个变量将会被加入到GraphKeys.TRAINABLE_VARIABLES集合。在Tensorflow中可以通过tf.trainable_variables函数得到所有需要优化的参数。Tensorflow中提供的神经网络优化算法会将GraphKeys.TRAINABLE_VARIABLES集合中的变量作为默认优化的对象。

### 神经网络优化算法
梯度下降算法和随机梯度下降算法。

### 神经网路进一步优化
Tensorflow提供一种灵活的学习率设置方法——指数衰减法。tf.train.exponential_decay函数实现了指数衰减学习率。

```
global_step = tf.Variable(0)

# 通过exponential_decay函数生成学习率
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

# 使用指数衰减学习率，在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也更新
learning_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
```
因为指定了staircase=True，所以每训练100轮后学习率乘以0.96。

### 正则化

```
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)

loss = tf.reduce_mean(tf.square(y_ - y)) + l2_regularizer(r)(w)
```
r是正则化项的参数。<br>
如果神经网络很复杂，参数增多，上面这种定义会导致损失函数loss的定义很长。更主要的是，当神经网络结构复杂之后定义网络结构部分和计算损失函数的部分可能不在同一个函数中，这样通过变量这种方式计算损失函数就很不方便。此时可以使用Tensorflow中提供的集合（collection）

```
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer


def get_weight(shape, r):
    var = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
    # add_to_collection函数将这个新生成的变量的L2正则化损失项加入集合
    # 这个函数的第一个参数'losses'是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection('losses', l2_regularizer(r)(var))


x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
batch_size = 8

layer_dimension = [2, 10, 10, 10, 1]
n_layer = len(layer_dimension)
cur_layer = x
in_dim = layer_dimension[0]

for i in range(1, n_layer):
    out_dim = layer_dimension[i]
    weigth = get_weight([in_dim, out_dim], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dim]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weigth) + bias)
    in_dim = out_dim

mse_loss = tf.reduce_mean(tf.square(y - cur_layer))
# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
```

### 滑动平均模型
**TODO**

### 变量管理
Tensorflow中通过变量名称获取变量的机制主要是通过tf.get_variable和tf.variable_scope函数实现的。

tf.get_variable用于创建变量时和tf.Variable的功能是基本等价的，但是前者的name参数是必填的参数。

```
v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
```
如果需要通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生成一个上下文管理器，并明确指定在这个上下文管理器中，tf.get_vairable将直接获取已经生成的变量

```
import tensorflow as tf

with tf.variable_scope('foo'):
    v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0, tf.float32))

with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v', [1])
    print(v == v1)
```

当tf.variable_scope函数使用reuse=True生成上下文管理器时，这个上下文管理器内所有的tf.get_variable函数会直接获取已经创建的变量，如果变量不存在，则tf.get_variable函数将报错；相反，如果tf.variable_scope函数使用参数reuse=None或者reuse=False创建上下文管理器，tf.get_variable操作将创建新的变量。

可以通过带命名空间名称的变量名来获取其它命名空间下的变量。

```
with tf.variable_scope('', reuse=True):
    v = tf.get_variable('foo/bar/v', [1])
``

### 模型持久化
通过tf.train.Saver类保存和还原一个神经网络（保存为.ckpt文件）。如果不希望重复定义图上的运算，也可以直接加载已经持久化的图。

```
import tensorflow as tf

saver = tf.train.import_meta_graph('./model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
```

如果只需要保存或者加载部分变量，比如保存一个6层神经网络的前5层变量，而重新训练第6层。为了保存或者加载部分变量，在声明tf.train.Saver类时可以提供一个列表来指定需要保存或者加的变量。

```
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')

# 原来名称为v1的变量现在加载到变量名'other-v1'
saver = tf.train.Saver({'v1': v1, 'v2': v2})
```

使用tf.train.Saver会保存运行TensorFlow程序所需要的全部信息。在只做测试或者离线预测的时候，就不需要获取这些细节。Tensorflow提供了convert_variables_to_constants函数，通过这个函数可以将计算图中的变量及其取值通过常量的方式保存，这样整个TensorFlow计算图可以统一存放在一个文件中。

```
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1], name='v1'))
v2 = tf.Variable(tf.constant(2.0, shape=[1], name='v2'))
result = v1 + v2

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将导出的模型存入文件
    with tf.gfile.GFile('./model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
```

通过以下程序可以直接计算定义的加法运算的结果。可以时间迁移学习

```
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    with gfile.FastGFile('./model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        result = tf.import_graph_def(graph_def, return_elements=['add:0'])
        print(sess.run(result)[0])
```
