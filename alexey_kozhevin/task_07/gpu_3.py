import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2, 1" 
# Numbers of GPUs that will be available. Numbers will be sorted and corresponding GPUs will be
# device:GPU:0, device:GPU:1, ...

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

a = tf.constant(3.0, name='a')
b = tf.constant(4.0, name='b')    
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run([a, b]))

# .........
# [name: "/device:CPU:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 6803028698056757120
# , name: "/device:GPU:0"
# device_type: "GPU"
# memory_limit: 224526336
# locality {
#   bus_id: 1
# }
# .........
# b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
# 2018-01-31 13:46:11.461710: I tensorflow/core/common_runtime/placer.cc:874] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
# a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
# 2018-01-31 13:46:11.461751: I tensorflow/core/common_runtime/placer.cc:874] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
# [3.0, 4.0]