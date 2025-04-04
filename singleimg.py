from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import json
import base64

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="包含导出模型的目录")
parser.add_argument("--input_file", help="输入 PNG 图像文件")
parser.add_argument("--output_file", help="输出 PNG 图像文件")
a = parser.parse_args()

def main():
    # 设置模型目录、输入文件和输出文件的默认值
    a.model_dir = "D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/exportmodel/r=2threepicture/"
    a.input_file = "D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/DAIStestdata/depth/r=2/prepcomup/data0119.png"
    a.output_file = "D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/prdeictdata/data0119.png"

    # 读取输入 PNG 图像文件的二进制数据
    with open(a.input_file, "rb") as f:
        input_data = f.read()

    # 创建包含 base64 编码输入数据的输入实例字典
    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))

    # 启动 TensorFlow 会话
    with tf.Session() as sess:
        # 找到模型目录中的最新检查点
        checkpoint_path = tf.train.latest_checkpoint(a.model_dir)
        # 导入模型元图并从检查点还原会话
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
        saver.restore(sess, checkpoint_path)
 
        input_vars = json.loads(tf.get_collection("inputs")[0].decode())
        output_vars = json.loads(tf.get_collection("outputs")[0].decode())
        input_tensor = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output_tensor = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

        # 将输入数据转换为 NumPy 数组并运行推断
        input_value = np.array(input_instance["input"])
        output_value = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(input_value, axis=0)})[0]

    # 创建包含 base64 编码输出数据的输出实例字典
    output_instance = dict(output=output_value.decode("ascii"), key="0")

    # 解码 base64 并将输出数据写入指定文件
    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

    with open(a.output_file, "wb") as f:
        f.write(output_data)

# 调用主函数
main()
