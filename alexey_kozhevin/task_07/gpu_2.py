import tensorflow as tf

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

device_count = {
    'CPU' : 1,
    'GPU' : 0
} # number of CPUs and GPUs to allow to use

with tf.Session(config=tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0}, log_device_placement=True)) as sess:
    a = tf.constant(3.0, name='a')
    b = tf.constant(4.0, name='b') 
    print(sess.run([a, b]))

# .........
# [name: "/device:CPU:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 6670400204173022778
# , name: "/device:GPU:0"
# device_type: "GPU"
# memory_limit: 7969613415
# locality {
#   bus_id: 1
# }
# incarnation: 6994509715148966106
# physical_device_desc: "device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1"
# , name: "/device:GPU:1"
# device_type: "GPU"
# memory_limit: 224526336
# locality {
#   bus_id: 1
# }
# .........
# b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# 2018-01-31 13:41:57.725715: I tensorflow/core/common_runtime/placer.cc:874] b: (Const)/job:localhost/replica:0/task:0/device:CPU:0
# a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# 2018-01-31 13:41:57.725751: I tensorflow/core/common_runtime/placer.cc:874] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
# [3.0, 4.0]