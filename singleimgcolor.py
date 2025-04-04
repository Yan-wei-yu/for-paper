import tensorflow as tf
import numpy as np
import json
import base64
import os


def run_inference(input_data, sess, input_tensor, output_tensor):
    """
    运行推断的核心函数。
    :param input_data: 输入的原始二进制数据
    :param model_dir: 模型所在目录
    :param sess: TensorFlow 会话
    :param input_tensor: 模型输入张量
    :param output_tensor: 模型输出张量
    :return: 推断后的输出数据
    """
    # 创建包含 base64 编码输入数据的输入实例字典
    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))

    # 将输入数据转换为 NumPy 数组并运行推断
    input_value = np.array(input_instance["input"])
    output_value = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(input_value, axis=0)})[0]

    # 创建包含 base64 编码输出数据的输出实例字典
    output_instance = dict(output=output_value.decode("ascii"), key="0")

    # 解码 base64 并返回输出数据
    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

    return output_data


def process_folder(input_dir, output_dir, model_dir):
    """
    遍历整个输入文件夹，对每张图片进行推断。
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param model_dir: 模型目录路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 启动 TensorFlow 会话
    with tf.Session() as sess:
        # 找到模型目录中的最新检查点
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        if not checkpoint_path:
            raise FileNotFoundError("无法找到检查点，请确认模型目录是否正确。")

        # 导入模型元图并从检查点还原会话
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
        saver.restore(sess, checkpoint_path)

        # 获取模型的输入和输出张量
        input_vars = json.loads(tf.get_collection("inputs")[0].decode())
        output_vars = json.loads(tf.get_collection("outputs")[0].decode())
        input_tensor = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output_tensor = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

        # 遍历输入文件夹中的每个文件
        for file_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # 确保处理的文件是图片
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"跳过非图片文件: {file_name}")
                continue

            print(f"处理文件: {file_name}")
            # 读取输入文件的原始二进制数据
            with open(input_path, "rb") as f:
                input_data = f.read()

            # 运行推断
            result = run_inference(input_data, sess, input_tensor, output_tensor)

            # 保存输出到文件
            with open(output_path, "wb") as f:
                f.write(result)

            print(f"已保存结果到: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/prdeictdata/fortestDAIS/r=2threepicture/", help="输入文件夹路径")
    parser.add_argument("--output_dir", default="D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/prdeictdata/forprdeictDAIS/r=2threepicture/",help="输出文件夹路径")
    parser.add_argument("--model_dir",default="D://Users//user//Desktop//weiyundontdelete//GANdata//trainingdepth//DAISdepth//alldata//exportmodel//r=2threepicture//", help="模型目录路径")
    args = parser.parse_args()

    # 处理整个文件夹
    process_folder(args.input_dir, args.output_dir, args.model_dir)
