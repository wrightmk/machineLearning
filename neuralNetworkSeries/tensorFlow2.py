# computation graph
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
# matmul for matrix multiplication
print(result)
# ---------------------
# model it in the session:

# sess = tf.Session()
# print(sess.run(result))
# sess.close()
# verbose form for (no need to close though)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)
