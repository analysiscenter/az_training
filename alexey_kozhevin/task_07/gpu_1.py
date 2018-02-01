import tensorflow as tf

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

with tf.device('/cpu:0'): # device name
    a = tf.constant(3.0, name='a')
with tf.device('/gpu:0'): # device name
    b = tf.constant(4.0, name='b')
c = a + b
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run([a, b, c]))

# .........
# [name: "/device:CPU:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 16166227091427687088
# , name: "/device:GPU:0"
# device_type: "GPU"
# memory_limit: 7969613415
# locality {
#   bus_id: 1
# }
# incarnation: 563639740333383096
# physical_device_desc: "device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1"
# , name: "/device:GPU:1"
# device_type: "GPU"
# memory_limit: 224526336
# locality {
#   bus_id: 1
# }
# .........
# add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
# 2018-01-31 13:50:05.824423: I tensorflow/core/common_runtime/placer.cc:874] add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
# b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
# 2018-01-31 13:50:05.824448: I tensorflow/core/common_runtime/placer.cc:874] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
# a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# 2018-01-31 13:50:05.824462: I tensorflow/core/common_runtime/placer.cc:874] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
# [3.0, 4.0, 7.0]