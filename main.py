import os

import tensorflow as tf

base_directory = '/home/oxvsys/'
image_directory = 'Desktop/tensorflow_test/test/'
location = base_directory + image_directory
all_images = os.listdir(location)
print(all_images)
categories = [line.rstrip() for line
              in tf.gfile.GFile("Results/names.txt")]


def recognize(image):
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    with tf.gfile.FastGFile("Results/graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print(image)
        output = categories[top_k[0]]
        print(output,100*predictions[0][top_k[0]])
        output = categories[top_k[1]]
        print(output,100*predictions[0][top_k[1]])
        # for node_id in top_k:
        #     human_string = categories[node_id]
        #     score = predictions[0][node_id]
        #     print(image)
        #     print('%s (score = %.5f)' % (human_string, score * 100))


for image in all_images:
    recognize(location + image)
