#use myself code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import cv2
# import historgramloss
# 參考

# https://colab.research.google.com/drive/182CGDnFxt08NmjCCTu5jDweUjn3jhB2y
parser = argparse.ArgumentParser()
#--input_dir:包含圖像的文件夾路徑。
parser.add_argument("--input_dir", help="path to folder containing images")
# which_direction #選項：train, test, export 運行模式。
parser.add_argument("--mode", choices=["train", "test", "export"])
# 輸出文件存放位置。
parser.add_argument("--output_dir", help="where to put output files")
#--seed:類型：int 說明：隨機種子
parser.add_argument("--seed", type=int)
#--checkpoint:說明：要恢復訓練或測試的檢查點目錄。 用途：指定提取特徵的檢查點目錄。
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")
#--cktCentralSul:說明：提取牙齒中央溝特徵的檢查點目錄。
parser.add_argument("--cktCentralSul", default=None,
                    help="directory with checkpoint to extract teeth central groove features")
#--max_steps:說明：訓練步數（設為0則禁用）。用途：限制訓練的最大步數。
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
#--max_epochs:類型：int說明：訓練輪數。用途：限制訓練的最大輪數。
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
#--summary_freq:類型：int默認值：100說明：更新和保存訓練過程的摘要信息。包括損失函數值、學習率、模型參數分佈等
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
#--progress_freq:類型：int默認值：50說明：用途：包括當前步數、損失值、訓練速度等即時信息。
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
#--trace_freq:類型：int默認值：0說明：包括每個操作的執行時間、內存使用等。跟蹤會顯著降低執行速度，所以默認值為0（即不跟蹤）。
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
#--display_freq:類型：int默認值：2000說明：每display_freq步寫當前訓練圖像。用途：設置圖像顯示的頻率。
parser.add_argument("--display_freq", type=int, default=5000,
                    help="write current training images every display_freq steps")
# --save_freq:類型：int默認值：2000說明：每save_freq步保存模型（設為0則禁用）。用途：設置模型保存的頻率。
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
#--aspect_ratio:類型：float默認值：1.0說明：輸出圖像的寬高比。用途：設置輸出圖像的寬高比。
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
#--lab_colorization:類型：布爾說明：將輸入圖像分為亮度（A）和顏色（B）。用途：啟用或禁用LAB顏色分離。
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
#--batch_size:類型：int默認值：1說明：批次中的圖像數量。用途：設置訓練批次的大小。
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
# --which_direction:類型：str默認值：AtoB選項：AtoB, BtoA說明：圖像轉換方向。用途：指定圖像轉換的方向。
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
# --ngf:類型：int默認值：64說明：第一個卷積層中生成器濾波器的數量。用途：設置生成器第一層的濾波器數量。
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
# --ndf:類型：int默認值：64說明：第一個卷積層中判別器濾波器的數量。用途：設置判別器第一層的濾波器數量。
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
# --nldf:類型：int默認值：128說明：第一個卷積層中局部判別器濾波器的數量。用途：設置局部判別器第一層的濾波器數量。
parser.add_argument("--nldf", type=int, default=128, help="number of local discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
#--scale_size:類型：int默認值：800說明：在裁剪到256x256之前將圖像縮放到此大小。用途：設置圖像縮放的大小。
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
# --flip 和 --no_flip:類型：布爾說明：水平翻轉圖像。用途：設置是否水平翻轉圖像。
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
#--lr:類型：float默認值：0.0002說明：Adam優化器的初始學習率。用途：設置學習率。
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
# --beta1:類型：float默認值：0.5說明：Adam優化器的動量項。用途：設置Adam優化器的動量。
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
#--l1_weight:類型：float默認值：100.0說明：生成器梯度的L1項權重。用途：設置L1損失的權重。
parser.add_argument("--per_weight", type=float, default=50.0, help="weight on per term for generator gradient")#50
# 感知損失 for 生成器
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--discrim_m", type=float, default=0.25, help="margin on GAN term for distrim percernal loss")
parser.add_argument("--dis_per_w", type=float, default=20.0, help="weight on GAN term for distrim percernal loss")#100
# parser.add_argument("--saveHide_freq", type=int, default=120000, help="保存隐藏层")
# 感知損失 for 鑑別器
# --gan_weight:類型：float默認值：1.0說明：生成器梯度的GAN項權重。用途：設置GAN損失的權重。
parser.add_argument("--cenSul_weight", type=float, default=50.0, help="weight on GAN term for central Sul loss")
parser.add_argument("--over_occlusion_weight", type=float, default=10.0, help="weight for over-occlusion loss")
parser.add_argument("--under_occlusion_weight", type=float, default=5.0, help="weight for under-occlusion loss")
# --cenSul_weight:類型：float默認值：100.0說明：中央溝損失的權重。用途：設置中央溝損失的權重。
parser.add_argument("--collision_weight", type=float, default=100.0, help="weight for collision loss")
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
parser.add_argument("--hist_weight", type=float, default=50.0, help="weight on GAN term for hist loss")
# parser.add_argument("--hist_weight", type=float, default=50.0, help="weight on GAN term for hist loss")


# --output_filetype:類型：str默認值：png選項：png, jpeg說明：輸出文件類型。用途：設置輸出文件的格式。

a = parser.parse_args()
# argparse模塊來解析命令行參數。所有在之前定義的參數都會被解析並存儲在a這個變量中。
EPS = 1e-12
CROP_SIZE = 256
# EPS: 一個非常小的數，用來避免數值計算中的除零錯誤。
# CROP_SIZE: 定義圖像裁剪的大小，這裡是256。
Examples = collections.namedtuple("Examples", "paths, inputs, condition1, condition2, targets, count, steps_per_epoch")
# Examples的命名元組，包含以下字段：
# paths: 圖像路徑列表。
# inputs: 輸入圖像。
# condition1: 條件1（可能是用於圖像轉換的某些條件）。
# condition2: 條件2（另一個條件）。
# targets: 目標圖像（真實圖像或轉換後的圖像）。
# count: 圖像的總數。
# steps_per_epoch: 每個epoch的步數。
# discrim_loss_per、gen_per_loss
# Model = collections.namedtuple("Model",
#                                "outputs, predict_real, predict_fake, global_discrim_loss,local_discrim_loss,discrim_loss_per,discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1,gen_per_loss,gen_loss_CenSul, gen_grads_and_vars, train")
# Model = collections.namedtuple("Model",
#                                "outputs,predict_local_real0,predict_local_fake0, predict_local_real1,predict_local_fake1, predict_local_real2,predict_local_fake2, predict_real, predict_fake, global_discrim_loss,local_discrim_loss,discrim_loss_per,discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1,gen_per_loss,histogram_loss,gen_loss_CenSul, gen_grads_and_vars, train")

Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, global_discrim_loss,local_discrim_loss,discrim_loss_per,discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1,gen_per_loss,histogram_loss,gen_loss_CenSul,gen_loss_collision, gen_grads_and_vars, train")
# Model的命名元組，包含以下字段：
# outputs: 生成的圖像輸出。
# predict_real: 對真實圖像的預測結果。
# predict_fake: 對假圖像的預測結果。
# global_discrim_loss: 全局判別器的損失。
# local_discrim_loss: 局部判別器的損失。
# discrim_grads_and_vars: 判別器的梯度和變量。
# gen_loss_GAN: 生成器的GAN損失。
# gen_loss_L1: 生成器的L1損失。
# gen_loss_CenSul: 生成器的中央溝損失。
# gen_grads_and_vars: 生成器的梯度和變量。
# train: 訓練操作。

# GAN中用於圖像數據的預處理
def preprocess(image):
    with tf.name_scope("preprocess"):
        # 圖像像素值從[0, 1]範圍轉換到[-1, 1]範圍。
        # [0, 1] => [-1, 1]
        return image * 2 - 1

# GAN中用於圖像數據的後處理
def deprocess(image):
    with tf.name_scope("deprocess"):
        # 圖像像素值從[-1, 1]範圍轉回[0, 1]範圍。
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

# 將顏色分為亮度 (L)、綠紅分量 (a)、和藍黃分量 (b)
# preprocess lab
def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        # 分解成 L、a 和 b 三個單獨的通道。
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan（亮度通道）：原始範圍是 [0, 100]，被轉換到 [-1, 1] 的範圍。
        # 計算過程：L_chan / 50 - 1，這樣 [0, 100] 會對應到 [-1, 1]。
        # a_chan 和 b_chan（色彩通道）：原始範圍大約是 [-110, 110]，也被縮放到 [-1, 1]。
        # # 計算過程：a_chan / 110 和 b_chan / 110
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

# 模型輸出的標準化數據轉回 Lab 顏色空間，恢復到原始的範圍，以便可視化或作為最終輸出。
# L_chan：逆轉換回 [0, 100] 的範圍。
# 計算過程：(L_chan + 1) / 2 * 100，這樣 [-1, 1] 會被轉換回 [0, 100]。
# a_chan 和 b_chan：逆轉換回大約 [-110, 110] 的範圍。
# 計算過程：a_chan * 110 和 b_chan * 110。
def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        #  最後使用 tf.stack 將三個通道重新合併成圖像
        # 這次的 axis=3 是因為要處理的是一個批次的圖像，通常形狀為 [batch_size, height, width, 3]
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


# L 通道：表示亮度（lightness），範圍從 0（黑色）到 100（白色）。
# a 通道：表示從綠色到紅色的色彩分佈，範圍約為 -110（綠色）到 110（紅色）。
# b 通道：表示從藍色到黃色的色彩分佈，範圍約為 -110（藍色）到 110（黃色）。
# 在將經過增強的亮度通道 (L channel) 與圖像的色彩通道 (a 和 b channels) 組合，然後將其轉換回 RGB 顏色空間。
# 目的是對圖像進行增強處理（如亮度調整），並將處理後的圖像輸出為 RGB 格式。
def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    # 這行代碼將 image 沿第 3 軸解壓，分解為單獨的 a 和 b 色彩通道。
    a_chan, b_chan = tf.unstack(image, axis=3)
    # 這行代碼將 brightness 沿第 3 軸壓縮，去除單通道的維度，得到 L 通道。
    L_chan = tf.squeeze(brightness, axis=3)
    # 使用 deprocess_lab 函數將 L、a 和 b 通道重新合併，並將其逆轉換回標準的 Lab 色彩空間範圍。
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    # 將 Lab 顏色空間的圖像轉換為 RGB 顏色空間，使其適合於可視化或模型輸出。
    rgb = lab_to_rgb(lab)
    return rgb


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        # 輸入張量的最後一個維度。這用來定義卷積核的輸入通道數。
        in_channels = batch_input.get_shape()[3]
        # 建立一個卷積核，形狀為 [4, 4, in_channels, out_channels]，
        # 其中 4x4 是卷積核的空間尺寸，in_channels 是輸入通道數，out_channels 是輸出的通道數。
        # 隨機初始化卷積核的權重。
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        # [0, 0]：對 batch 維度不進行填充。
        # [1, 1]：對高度維度每側填充 1。
        # [1, 1]：對寬度維度每側填充 1。
        # [0, 0]：對通道維度不進行填充。
        # mode=CONSTANT指定了填充的模式
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        # [1, stride, stride, 1] 表示卷積運行的步長，
        # 其中 1 表示 batch 和 channels 維度不進行步進，只在高度和寬度維度以 stride 進行步進。
        # adding="VALID" 表示沒有額外的填充，這意味著輸出尺寸會因卷積核減小。
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

# Leaky ReLU（Leaky Rectified Linear Unit）激活函數
def lrelu(x, a):
    # 並使用 a 作為負斜率。Leaky ReLU 是 ReLU 的變體，
    # 在輸入值為負時允許一些負輸出，避免 ReLU 出現「死亡」神經元問題。
    # 這樣在輸入值小於零時，激活函數仍會有梯度，從而能夠幫助網絡學習。
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        # 為了確保 x 被視為計算圖中的一個單獨節點
        x = tf.identity(x)
        # 0.5 * (1 + a) * x：這是輸入值 x 的線性部分。
        # 0.5 * (1 - a) * tf.abs(x)：這是輸入值絕對值的部分，用於控制當輸入為負時的輸出。
        # 在正數區域，Leaky ReLU 和 ReLU 表現一樣，在負數區域，輸出是 a * x，這個 a 是設置的負斜率。
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

# 批歸一化（Batch Normalization）操作，它有助於加速訓練速度和提高神經網絡的穩定性。

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        #創建了一個與輸入 input 一樣的張量，並在計算圖中增加了一個節點
        input = tf.identity(input)
        # 獲取輸入張量中通道的數量
        channels = input.get_shape()[3]
        # 對應於批歸一化後的平移項，初始化為全零。
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        # 對應於批歸一化後的縮放項，初始值為服從均值為 1.0，標準差為 0.02 的正態分佈。
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        # 計算 input 在 [0, 1, 2]（即 batch, height, width 維度）上的均值和方差，這樣每個通道會單獨計算均值和方差。
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        # 是一個小的常數，用於防止在歸一化時除以零。
        variance_epsilon = 1e-5
        # 將 input 正規化，使其具有零均值和單位方差，然後應用 scale 和 offset
        # normalized = scale * ((input - mean) / sqrt(variance + variance_epsilon)) + offset
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

# 實現反卷積
# batch_input:
# 這是輸入張量，形狀為 [batch, in_height, in_width, in_channels]。
# 代表多個樣本的批次，包含高度和寬度的特徵圖以及通道數。
# out_channels:
# 轉置卷積操作的輸出通道數（目標特徵圖的通道數）。
def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        # 輸入張量，形狀為 [batch, in_height, in_width, in_channels]。
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        # tf.get_variable():用於創建或重用變數，確保命名一致，便於共享。
        # name="filter": 卷積核的名稱。
        # [4, 4, out_channels, in_channels]: 卷積核的形狀，解釋如下：
        # 4, 4: 卷積核的高和寬（即 4x4）。
        # out_channels: 輸出的通道數。
        # in_channels: 輸入的通道數。
        # dtype=tf.float32: 資料型別為 32 位浮點數。
        # initializer=tf.random_normal_initializer(0, 0.02):
        # 用均值為 0、標準差為 0.02 的正態分佈初始化卷積核權重。
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        # tf.nn.conv2d_transpose:TensorFlow 提供的轉置卷積函數。將特徵圖從較小的分辨率上採樣到較大的分辨率。
        # [batch, in_height * 2, in_width * 2, out_channels]:輸出形狀，特徵圖的高度和寬度是輸入的兩倍，通道數是 out_channels。
        # [1, 2, 2, 1]: 步長（stride），表示在每個空間維度上步長為 2（即上採樣倍數為 2），批次和通道不變（步長為 1）。
        # 使用 SAME 填充，輸出大小計算為： out_size=in_size×stride
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        # 返回計算後的張量 conv
        return conv


def check_image(image):
    # 取得張量 image 的最後一維的大小（即顏色通道數）。
    # 確保最後一維大小等於 1。若不滿足條件，將觸發錯誤並輸出自訂訊息
    assertion = tf.assert_equal(tf.shape(image)[-1], 1, message="image must have 1 color channels")
    # 確保在執行 tf.identity(image)（返回 image 本身）之前，執行 assertion 的檢查。
    # 如果檢查失敗，程式會報錯並停止執行
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)
    # image.get_shape().ndims:
    # 獲取張量 image 的靜態維度數（在編譯時可確定的維度數）。
    # 如果 image 的維度數不在 3（單張影像，如 [height, width, channels]）
    # 或 4（影像批次，如 [batch_size, height, width, channels]）之間，將拋出錯誤。
    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # image.get_shape() 返回靜態形狀（如果可能）。將其轉換為列表以進行修改。
    # 將最後一維設置為 1 並修改
    shape = list(image.get_shape())
    shape[-1] = 1
    image.set_shape(shape)
    return image

# RGB 到 Lab 色彩空間
#  RGB 色彩空間的影像轉換為 CIELAB 色彩空間。Lab 是一種感知均勻的色彩空間，適合用於影像處理或機器學習應用。
# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        # 將影像展平成形狀為 [num_pixels, 3] 的張量，每行代表一個像素的 RGB 值。
        # tf.reshape 根據給定的形狀，按照數據的存儲順序重新組織數據。
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        # RGB 到 XYZ 轉換，將 sRGB 轉為線性 RGB
        with tf.name_scope("srgb_to_xyz"):
            # 根據 sRGB 的特性，低於 0.04045 的值需用不同公式轉換。 tf.cast改變張量數據類型 
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                        ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            # 將非線性的 sRGB 值轉為線性 RGB 值。
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)
            # D65 白點 是一種常用的標準光源，模擬日光的光譜分佈。對應的標準白點值是：
            # Xn = 0.950456
            # Yn = 1.0
            # Zn = 1.088754
            # 將 XYZ 正規化到 D65 白點：
            # tf.multiply這是一個逐元素乘法操作
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])
            # 將正規化 XYZ 值映射到 Lab 的非線性空間：
            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                        xyz_normalized_pixels ** (1 / 3)) * exponential_mask
            # tf.constant 的設計是為了與 TensorFlow 的計算圖機制和數據管理兼容
            # 而直接用 Python 的 array 雖然簡單
            # 但在 TensorFlow 操作中不夠高效，也不方便進行數據類型管理或跨設備運算
            # 將 fx, fy, fz 映射到 Lab：
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
        # 將轉換後的像素重新 reshape 為輸入形狀。
        return tf.reshape(lab_pixels, tf.shape(srgb))
    
# CIE LAB 色彩空間轉換為 sRGB 色彩空間
#  Lab 到 RGB 色彩空間
# 這段程式完成了 LAB -> XYZ -> RGB -> sRGB 的轉換。
# 該過程在色彩處理領域中常用，特別是用於圖像顯示和編輯。
# LAB 色彩空間接近於人類視覺的感知，轉換為 sRGB 後可以用於螢幕顯示。
def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # 將 LAB 轉換為 XYZ（CIE XYZ 色彩空間）：
            # 透過矩陣相乘來實現。
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            # 轉換為實際的 XYZ 色彩值：
            #  𝑓(𝑋)>𝜖，用 𝑓(𝑋)^3計算。
            #  𝑓(𝑋)=<𝜖，用 線性計算
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                        fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            # XYZ 色彩值進行 D65 白點的反正規化
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            # 使用標準的 XYZ-to-RGB 矩陣將 XYZ 色彩空間轉換為線性 RGB。
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            # 任何小於clip_value_min的值都被設定為clip_value_min。任何大於clip_value_max的值都會設定為clip_value_max。
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            # 將線性 RGB 轉換為標準的 sRGB（伽瑪校正）：
            # 當 𝑅𝐺𝐵≤0.0031308，使用線性公式
            # 當 𝑅𝐺𝐵>0.0031308，使用指數公式。
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                        (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask
        # 將扁平化的 RGB 結果重新調整為與輸入圖像相同的形狀。
        return tf.reshape(srgb_pixels, tf.shape(lab))


# load dataset
def load_examples():
    # 從指定目錄中讀取 .jpg 或 .png 文件
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # 如果所有圖像文件名是數字，則按數值排序；否則按字典序排序。
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    # load images
    with tf.name_scope("load_images"):
        # tf.train.string_input_producer 創建文件路徑隊列，並通過 tf.WholeFileReader 讀取文件。
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        # 將圖像數據解碼為張量，並轉換為 tf.float32（範圍 [0,1][0,1]）。
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        #         確保圖像是單通道（如灰度圖），否則觸發錯誤。
        # 使用 tf.set_shape 明確設定圖像的形狀。
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 1, message="image does not have 1 channels")
        #print(tf.shape(raw_input)[2])
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 1])
        # 如果使用 LAB 色彩空間，將 RGB 圖像轉換為 LAB，並提取亮度和顏色通道。
        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # 否則，將圖像分割為多個區域，並標準化到範圍 [−1,1][−1,1]。分割圖像
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            #a_images = preprocess(raw_input[:, :width // 4, :])
            #c_image1 = preprocess(raw_input[:, :width // 4, :])
            #c_image2 = preprocess(raw_input[:, :width // 4, :])
            #b_images = preprocess(raw_input[:, width // 4:, :])
            # 假設圖像水平拼接了 4 個區域，按列（axis=1）分割。
            # 每個區域進行標準化預處理（可能將值縮放到範圍 [−1,1][−1,1]）。
            a_imagesT, c_image1T, c_image2T, b_imagesT=tf.split(raw_input, 4, axis=1)
            a_images = preprocess(a_imagesT);
            c_image1 = preprocess(c_image1T);
            c_image2 = preprocess(c_image2T);
            b_images = preprocess(b_imagesT);
    # 設定決定輸入與輸出的方向：
    if a.which_direction == "AtoB":
        inputs, condit1, condit2, targets = [a_images, c_image1, c_image2,b_images]
    elif a.which_direction == "BtoA":
        inputs, condit1, condit2, targets = [b_images, c_image2, c_image1, a_images]
    else:
        raise Exception("invalid direction")

    # 確保輸入與輸出的隨機操作（如裁剪、翻轉等）保持一致。
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        # 隨機水平翻轉：根據設定隨機翻轉圖像。
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        # 調整圖像大小：使用區域插值法將圖像調整到指定大小。
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        # 使用 tf.random.uniform 在範圍 [0, a.scale_size - CROP_SIZE + 1) 生成一個隨機偏移量。
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        # 如果 a.scale_size > CROP_SIZE，執行裁剪，從圖像中隨機取出一個大小為 𝐶𝑅𝑂𝑃_𝑆𝐼𝑍𝐸×𝐶𝑅𝑂𝑃_𝑆𝐼𝑍𝐸 的區域。
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        # 則拋出異常，因為圖像不應小於裁剪大小
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        # 返回經過翻轉、調整大小和裁剪的圖像。
        return r

    # def transform(image):
    #     r = image
    #     # 隨機水平翻轉（可選）
    #     if a.flip:
    #         r = tf.image.random_flip_left_right(r, seed=seed)

    #     # 調整圖像大小至 scale_size
    #     r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

    #     # 計算整數的 offset_height 和 offset_width
    #     offset_height = int(95 * (a.scale_size / CROP_SIZE))
    #     offset_width = int(80 * (a.scale_size / CROP_SIZE))

    #     # 裁剪放大後的區域
    #     r = tf.image.crop_to_bounding_box(
    #         r,
    #         offset_height=offset_height,
    #         offset_width=offset_width,
    #         target_height=CROP_SIZE,
    #         target_width=CROP_SIZE
    #     )

    #     return r

    # 對輸入 (inputs)、條件圖像 (condit1 和 condit2)、目標圖像 (targets) 分別應用 transform 函數進行預處理
    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("con_image1"):
        con_image1 = transform(condit1)

    with tf.name_scope("con_image2"):
        con_image2 = transform(condit2)

    with tf.name_scope("target_images"):
        target_images = transform(targets)
    # tf.train.batch：
    # 將處理過的圖像和對應的文件路徑打包成批次。
    paths_batch, inputs_batch, con1_batch, con2_batch, targets_batch\
        = tf.train.batch([paths, input_images, con_image1, con_image2, target_images],  batch_size=a.batch_size)
    #     steps_per_epoch：每個 epoch 的訓練步數。
    # len(input_paths)：總圖像數量。
    # a.batch_size：每個批次的圖像數量。
    # 使用 math.ceil 確保即使數據量不是批次大小的整數倍，最後一個不完整的批次也會被計入。
    
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))
    # Examples 對象：
    # 一個封裝數據集的結構體或類（假設已在代碼其他部分定義）。
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        condition1=con1_batch,
        condition2=con2_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )
def create_generator_groove(generator_inputs, generator_outputs_channels, condition=None):
    # layers：存儲生成器模型的所有層，從編碼器到解碼器。
    layers = []
    # 編碼器通過逐層下採樣將輸入的特徵圖尺寸減小，通道數逐漸增加。
    # 第一層單獨實現，後續層根據 layer_specs 自動生成。
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    if condition is not None:
        generator_inputs = tf.concat([generator_inputs, condition], axis=3)
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)
    # 這些層會依次對輸入特徵圖進行多次下採樣。
    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    # 每層都通過 LeakyReLU 激活函數進行處理，並使用卷積層將特徵圖進行下採樣。
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
        # 每次卷積後應用批量正規化（batch normalization）來加速訓練並提高穩定性。
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
        # 將每一層的輸出添加到 layers 列表中。
            layers.append(output)
    # 定義解碼器部分的層配置，包括每層的輸出通道數以及使用的 dropout 比例（0.0 表示無 dropout）。
    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
    # 在每個解碼器層中，根據是否需要跳躍連接（skip connection），將上一層的輸出與相應編碼器層的輸出進行拼接。
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            # 使用反卷積層（deconv）進行上採樣，將圖像尺寸擴大並減少通道數。
            output = deconv(rectified, out_channels)
            output = batchnorm(output)
            # 根據配置的 dropout 比例，隨機丟棄部分單元來防止過擬合。
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
    # 最後一層解碼器將來自編碼器的第一層和最後一層的特徵圖進行拼接（跳躍連接）。
    # 使用反卷積層進行上採樣，將圖像尺寸擴展至原來的大小（256x256）。
    # 最後，使用 tanh 函數來進行輸出，這樣可以將輸出的值限制在 [-1, 1] 範圍內。
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)
    # 最後返回生成器的輸出，即 layers 列表中的最後一個層，這是經過解碼器處理後的最終圖像。
    return layers[-1]

def create_generator(generator_inputs, discrimCon1, discrimCon2, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    # 這行程式碼的作用是將三個張量（generator_inputs, discrimCon1, discrimCon2）沿著最後一個維度（通常是通道數維度，即 axis=3）進行拼接（concatenate）
    # 。具體來說，這個操作的意圖是將來自不同來源的特徵圖合併為一個單一的張量，
    generator_inputss=tf.concat([generator_inputs, discrimCon1, discrimCon2], axis=3)
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputss, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


# create model
def create_model(inputs, condition1, condition2, targets):
    def create_discriminator(discrim_inputs,discrim_con1, discrim_con2,  discrim_targets):
        n_layers = 4
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # tf.concat 在通道維度（axis=3）進行拼接，將判別器輸入、條件和目標圖像組合在一起。
        input = tf.concat([discrim_inputs, discrim_con1, discrim_con2, discrim_targets], axis=3)
        # input = tf.concat([discrim_inputs,discrim_targets], axis=3)
        # 第一層（layer_1）：輸入尺寸為 [batch, 256, 256, in_channels * 2]，
        # 經過卷積後尺寸縮小為 [batch, 128, 128, ndf]，並使用 Leaky ReLU 激活函數（lrelu）。
        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            # perlayers.append(rectified)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        # n_layers：設置判別器的卷積層數量，這裡定義為 3 層。
        # 接下來是三層卷積層（layer_2, layer_3, layer_4），每一層的輸出通道數會逐漸增大（ndf, ndf*2, ndf*4, ndf*8）
        # ，並且每層的步幅（stride）會根據需要設置為 2（除非是最後一層，步幅為 1）。
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)
        # 最後一層（layer_5）將卷積的輸出映射到 1 通道（即二分類結果：真或假）
        # 並使用 Sigmoid 函數將結果限制在 [0, 1] 範圍內，表示判別結果。
        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)
        # perlayers：這個列表包含了每一層的輸出結果。它包括所有中間層（rectified 激活後的層），以及最終輸出層
        return layers
    # perceTarget：目標圖像的特徵表示，通常是從一個預訓練的網絡中提取的特徵圖。
    # perceOutput：生成圖像的特徵表示，通常是生成器輸出的圖像的特徵圖。
    def perceptual_Loss(perceTarget, perceOutput):
        # 設定每層權重
        weights = [1.0, 2.0, 2.0]
        perLoss = 0.0
        for i in range(len(perceTarget)-2):
            # Calculate the size of the feature map
            C, H, W = tf.shape(perceTarget[i])[1], tf.shape(perceTarget[i])[2], tf.shape(perceTarget[i])[3]
            normalization_factor = tf.cast(C * H * W, tf.float32)
            # Compute the mean absolute difference normalized by feature map size and weighted by lambda
            loss = weights[i] * tf.reduce_sum(tf.abs(perceTarget[i] - perceOutput[i])) / normalization_factor
            perLoss += loss
        return perLoss
    
    
    # gan local discriminator
    # 主要關注圖像的特定區域（ROI）
    # discrim_inputs: 判別器的輸入，通常是生成器生成的圖像。
    # discrim_targets: 判別器的目標，通常是真實的圖像。
    # 起始點為 (80, 80)，表示從圖像的第 80 行、第 80 列開始。
    # 裁剪的大小為 (128, 128)，表示裁剪後的區域是 128x128 的正方形。
    def create_local_discriminator(discrim_inputs,discrim_con1, discrim_con2, discrim_targets):
        n_layers = 3
        layers = []

        #tensor ROI区域裁剪
        crop_inputs=tf.image.crop_to_bounding_box(discrim_inputs,80,80,128,128)
        crop_discrim_con1=tf.image.crop_to_bounding_box(discrim_con1,80,80,128,128)
        crop_discrim_con2 = tf.image.crop_to_bounding_box(discrim_con2, 80, 80, 128, 128)
        crop_targets = tf.image.crop_to_bounding_box(discrim_targets, 80, 80, 128, 128)

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # 將裁剪的輸入（crop_inputs）和目標（crop_targets）組合為一個張量。
        # 如果每個張量的通道數為 C，拼接後的通道數將為 2C。   
        input = tf.concat([crop_inputs,crop_discrim_con1,crop_discrim_con2, crop_targets], axis=3)
        # 輸入：拼接後的張量。
        # 輸出通道數：a.nldf（局部判別器的基礎通道數）。
        # 步幅：stride=2，表示每次移動兩個像素，用於下採樣。
        #  128x128x2C 下採樣為 64x64xC'。
        # layer_1: [batch, 128, 128, in_channels * 2] => [batch, 64, 64, nldf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.nldf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 64, 64, ndf ] => [batch, 32, 32, ndf * 2]
        # layer_3: [batch, 32, 32, ndf * 2] => [batch, 31, 31, ndf * 4]
        # 疊加判別器的中間卷積層，每層執行以下步驟：
        # 第一層輸出通道數為 a.nldf * 2。
        # 第二層輸出通道數為 a.nldf * 4。
        # 如果超過 8 倍的基礎通道數，將通道數固定為 a.nldf * 8。
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.nldf * min(2 ** (i + 1), 4)
                # 當 i == n_layers - 1 時，使用 stride=1，否則使用 stride=2。
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)
        # 最後一層不進行下採樣，以保留更多特徵信息。
        # 輸出通道數設為 1，即最後輸出是一個單通道的特徵圖。
        # 步幅設為 1，不進行下採樣。

        # layer_4: [batch, 31, 31, ndf * 4] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            # sigmoid 激活函數，將輸出值壓縮到 [0, 1] 區間，用於判別器的輸出
            output = tf.sigmoid(convolved)
            layers.append(output)
        # 返回 layers 列表中的最後一層（輸出張量），表示局部判別器的最終判定結果。
        return layers

    # with tf.name_scope 用來管理計算圖中的命名空間，從而使代碼的結構更清晰，便於調試和查找變量。
    #  TensorBoard 中區分或查看一組相關操作（如損失計算、可視化輸出等）的情況。
    # with tf.variable_scope 用於對**變量（variables）**進行命名和管理，特別是共享變量的場景。 支持變量重用
    # tf.variable_scope 用於共享變量（如生成器和判別器的參數），避免多次創建相同的變量。
    # gan generator
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, condition1, condition2, out_channels)
    # 全局判別器：從整體判斷真實性，比如整體輪廓和結構。
    # 局部判別器：專注於細節處理，比如紋理或局部一致性。
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            # 一次判別真實圖像 (predict_real)。
            predict_real = create_discriminator(inputs, condition1, condition2, targets)
            
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            # 一次判別生成的假圖像 (predict_fake)。(通過 reuse=True 共享參數)，
            predict_fake = create_discriminator(inputs, condition1, condition2, outputs)

            # 提取特定的 ROI (感興趣區域，如 128x128)，針對小範圍判斷真實性。   
    with tf.name_scope("real_local_discriminator"):
        with tf.variable_scope("local_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_local_real = create_local_discriminator(inputs, condition1, condition2, targets)

    with tf.name_scope("fake_local_discriminator"):
        # Setting reuse=True avoids creating new variables and reuses the ones from the local  real discriminator.
        with tf.variable_scope("local_discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_local_fake = create_local_discriminator(inputs, condition1, condition2, outputs)


    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # 引入感知層面的判別。
        discrim_loss_per = tf.nn.relu(tf.subtract(a.discrim_m,perceptual_Loss(predict_local_real,predict_local_fake)))
        # text{global_discrim_loss} = -/log(D(/text{real})) - /log(1 - D(/text{fake}))
        # 對真實圖像輸出值靠近 1，對假圖像輸出值靠近 0。
        # 判別器損失
        global_discrim_loss = tf.reduce_mean(predict_real[-1] + EPS) - tf.reduce_mean(predict_fake[-1] + EPS)
        # 判別局部特徵是否真實。
        # local_discrim_loss=tf.reduce_mean(-(tf.log(predict_local_real[-1] + EPS) + tf.log(1 - predict_local_fake[-1] + EPS)))
        local_discrim_loss=tf.reduce_mean((predict_local_real[-1] + EPS) -tf.reduce_mean( predict_local_fake[-1] + EPS))

        discrim_loss =global_discrim_loss+local_discrim_loss+discrim_loss_per*a.dis_per_w
        
        # discrim_loss =global_discrim_loss+local_discrim_loss

    #flyadd 构建中央沟提取模型
    with tf.name_scope("tarCentralSul_loss"):
        with tf.variable_scope("genTeethGroove"):
            cenSulTarget = create_generator_groove(targets,1)
    with tf.name_scope("outCentralSul_loss"):
        with tf.variable_scope("genTeethGroove", reuse=True):
            cenSulOutput = create_generator_groove(outputs,1)
    #flyadd


    with tf.name_scope("generator_loss"):
        # # 先把這幾張圖都歸一到同一範圍
        # # 先正規化到 [0,1]
        # outputs_norm = (outputs + 1.) * 0.5
        # condition1_norm = (condition1 + 1.) * 0.5
        # targets_norm = (targets + 1.) * 0.5

        # # 計算簽名差異
        # diff_gen_cond = outputs_norm - condition1_norm  # [batch, H, W, C]
        # diff_cond_tgt = condition1_norm - targets_norm  # [batch, H, W, C]

  
        # # 總碰撞損失：生成圖像與目標圖像的碰撞特徵差異
        # gen_loss_collision = tf.reduce_mean(tf.abs(diff_gen_cond - diff_cond_tgt)) 
        # 正規化到 [0, 1]
        outputs_norm = (outputs + 1.) * 0.5
        condition1_norm = (condition1 + 1.) * 0.5
        targets_norm = (targets + 1.) * 0.5

        # 計算差異
        diff_gen_cond = outputs_norm - condition1_norm  # [batch, H, W, C]
        diff_cond_tgt = targets_norm - condition1_norm  # [batch, H, W, C]

        # 為正負偏差設置不同權重
        positive_diff_gen = tf.nn.relu(diff_gen_cond)
        negative_diff_gen = tf.nn.relu(-diff_gen_cond)
        positive_diff_tgt = tf.nn.relu(diff_cond_tgt)
        negative_diff_tgt = tf.nn.relu(-diff_cond_tgt)

        # 計算加權損失
        collision_loss = (a.over_occlusion_weight * tf.reduce_mean(positive_diff_gen) +
                        a.under_occlusion_weight * tf.reduce_mean(negative_diff_gen))
        target_collision_loss = (a.over_occlusion_weight * tf.reduce_mean(positive_diff_tgt) +
                                a.under_occlusion_weight * tf.reduce_mean(negative_diff_tgt))

        # 碰撞損失
        gen_loss_collision = tf.abs(collision_loss - target_collision_loss)
        # 看起來這邊還要加值方圖損失函式在L1那邊
        # GAN Loss=−log(D(G(z)))
        # -log(predict_fake)，當 predict_fake 趨近 1 時，損失會接近 0。
        gen_loss_GAN = tf.reduce_mean((predict_fake[-1]))#1
        #  (outputs) 接近目標圖像 (targets)，提高生成結果的真實性。
        # L1 損失對像素值差異的敏感性較低，通常比 L2 損失更適合圖像生成。
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))#2
        # # 比較生成的圖像和真實圖像在特徵空間中的差異 (不是像素層面的直接差異)。
        # # 用法：提高生成圖像的高層次感知相似性，例如紋理或內容的相似度。
        # tf.nn.relu(tf.subtract(a.discrim_m,perceptual_Loss(predict_local_real,predict_local_fake)))
        gen_per_loss=perceptual_Loss(predict_local_real,predict_local_fake)#3
        # # 作用：專注於生成器對目標的特定區域 (如中央溝) 的生成質量，確保這部分的準確性。
        gen_loss_CenSul=tf.reduce_mean(tf.abs(cenSulTarget - cenSulOutput))#4
        # 作用：對生成圖像特徵的分佈與條件特徵 (condition2) 進行比較，確保生成的圖像符合指定的條件分佈。
        hist_fake = tf.histogram_fixed_width(predict_fake[-1],  [0.0, 255.0], 256)
        hist_real = tf.histogram_fixed_width(predict_real[-1],  [0.0, 255.0], 256)

        histogram_loss = tf.reduce_mean(
        tf.divide(
            tf.square(tf.cast(hist_fake, tf.float32) - tf.cast(hist_real, tf.float32)),
            tf.maximum(1.0, tf.cast(hist_real, tf.float32))  # 確保類型一致
        )
    )

        # 將上述多個損失以權重加權求和，平衡不同損失的影響。
        # gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight+gen_loss_CenSul*a.cenSul_weight+gen_per_loss * a.per_weight+histogram_loss * a.hist_weight
        # 結合所有損失，加權求和
        gen_loss = (gen_loss_GAN * a.gan_weight +
                    gen_loss_L1 * a.l1_weight +
                    gen_loss_CenSul * a.cenSul_weight +
                    gen_per_loss * a.per_weight +
                    histogram_loss * a.hist_weight +
                    gen_loss_collision * a.collision_weight)

    # 作用：使用 Adam 優化器更新與判別器相關的參數，讓其學習如何更好地區分真實與生成圖像。
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
    # 保證在每次生成器更新前，先完成判別器參數的更新，實現交替訓練。
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            # 使用 Adam 優化器更新與生成器相關的參數，提升生成器生成高質量圖像的能力。
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)


    # 指標的滑動平均值計算，用於平滑損失曲線，讓訓練過程中的指標更加穩定
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    #  gen_per_loss ,discrim_loss_per,
    # update_losses = ema.apply([global_discrim_loss,discrim_loss_per,local_discrim_loss, gen_loss_GAN, gen_loss_L1,gen_loss_CenSul, gen_per_loss,histogram_loss])
    update_losses = ema.apply([global_discrim_loss,discrim_loss_per,local_discrim_loss, gen_loss_GAN, gen_loss_L1,gen_loss_CenSul, gen_per_loss,histogram_loss,gen_loss_collision])
    # 管理訓練步驟，global_step 是 TensorFlow 內建變量，用於記錄當前訓練進行的步數。
    global_step = tf.contrib.framework.get_or_create_global_step()
    # a.lr = tf.train.exponential_decay(0.0001, global_step, decay_steps=10000, decay_rate=0.96, staircase=True)
    incr_global_step = tf.assign(global_step, global_step + 1)
    # train=tf.group(update_losses, incr_global_step, gen_train) 
    # 將指標更新、步驟遞增和生成器訓練綁定在一起，形成完整的訓練步驟。
    def mean_hide_layer(tensorlayer):
            tensordis=tensorlayer.get_shape().as_list()
            numdis=len(tensordis)
            lastdis=tensordis[numdis-1]
            layers=[]
            for i in range(lastdis):
                if i==0:
                    layers.append(tf.slice(tensorlayer, [0, 0, 0, i], [1, -1, -1, 1]))
                else:
                    templayer= layers[-1]+tf.slice(tensorlayer, [0, 0, 0, i], [1, -1, -1, 1])
                    layers.append(templayer)
            return layers[-1]/lastdis
    return Model(
        # predict_real=cenSulTarget,
        # predict_fake=cenSulOutput,
        # predict_local_real0=predict_local_real[0],
        # predict_local_fake0=predict_local_fake[0],
        # predict_local_real1=predict_local_real[1],
        # predict_local_fake1=predict_local_fake[1],
        # predict_local_real2=predict_local_real[2],
        # predict_local_fake2=predict_local_fake[2],
        predict_real=predict_real[-1],
        predict_fake=predict_fake[-1],
        global_discrim_loss=ema.average(global_discrim_loss),
        local_discrim_loss=ema.average(local_discrim_loss),
        discrim_loss_per=ema.average(discrim_loss_per),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_CenSul=ema.average(gen_loss_CenSul),
        gen_per_loss=ema.average(gen_per_loss),
        histogram_loss=ema.average(histogram_loss),
        gen_loss_collision=ema.average(gen_loss_collision),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

# 存儲「輸入 (inputs)」、「輸出 (outputs)」和「目標 (targets)」。
def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        # 創建保存影像的資料夾
        os.makedirs(image_dir)

    filesets = []
    # 遍歷批次中的所有圖像，保存至 PNG 文件
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        # 返回文件集列表，記錄每個保存文件的相關信息：
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    # 生成或更新一個 HTML 文件，用於以表格形式展示保存的圖像，方便直觀查看輸入、輸出與目標的比較。
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path



# a.cktCentralSul：指定一個路徑，可能用於讀取特定模型或數據文件。
# a.input_dir：訓練數據的輸入目錄。
# a.mode = "train"：設定目前運行模式為訓練。
# a.output_dir：訓練輸出的結果（例如，生成的模型檔案）將存儲到這個目錄。
# a.max_epochs：設置訓練的最大 epoch 數，這裡是 400。
# a.which_direction：設定轉換方向（BtoA 表示從 B 映射到 A）。
def main():
    #     if tf.__version__.split('.')[0] != "1":
    #         raise Exception("Tensorflow version 1 required")

    # # # # 训练的时候的参数(由于采用
    a.cktCentralSul = "D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/DAISgroove/"

    # # # # # # # # # # 训练的时候的参数(由于采用
    # # a.input_dir = 'D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/depthfordifferentr/DAISdepth/bb/r=2/final'
    # a.input_dir = 'D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/depthfordifferentr/DCPRdepth/bb/r=1/final'
    a.input_dir = "D:/Users/user/Desktop/weiyundontdelete/GANdata/forjournal/inlayonlay/train/final/"
    a.mode = "train"
    a.output_dir = "D:/Users/user/Desktop/weiyundontdelete/GANdata/forjournal/inlayonlay/train/trainmodel/"
    a.max_epochs=400
    a.which_direction = "BtoA"

    # a.checkpoint = "D:/Users/user/Desktop/weiyundontdelete/GANdata/trainingdepth/DAISdepth/alldata/model/DCPRdepth=1collisionandchangehistorgram/"
    # a.mode = "export"
    # a.output_dir ="D://Users//user//Desktop//weiyundontdelete//GANdata//trainingdepth//DAISdepth//alldata//exportmodel//DCPRdepth=1collisionandchangehistorgramr//"
    # a.which_direction = "BtoA"

    # 测试的时候的参数
    #a.input_dir = "D:/Tensorflow/DAIS/test"
    #a.mode = "test"
    #a.output_dir = "D:/Tensorflow/DAIS/test_result"
    #a.checkpoint = "D:/Tensorflow/DAIS/Checkpoint"
    # 下面這句程序不用添加：因爲在checkpoint中已經把包含了 BtoA的option
    #  options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    #     a.which_direction = "BtoA"

    #     python pix2pix.py /
    #   --mode test /
    #   --output_dir facades_test /
    #   --input_dir facades/val /
    #   --checkpoint facades_train

    #  為隨機數生成器設置種子，確保結果可重現。
    # 如果未手動指定種子（a.seed），會隨機生成一個種子。
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    # 输出目录设置
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")
        # 讀取檢查點文件中的 options.json 配置。
        # options 中的參數（例如 which_direction, ngf, ndf 等）會根據檢查點的內容進行加載。
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # 測試模式下禁用圖像翻轉和縮放等數據增強功能，確保測試結果不受隨機性影響。
        a.scale_size = CROP_SIZE
        a.flip = False
    # 列出當前運行時的所有參數，方便檢查設置是否正確。
    for k, v in a._get_kwargs():
        print(k, "=", v)
    # 將所有參數保存到 options.json 文件，方便日後檢查或複現訓練過程。
    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # 模型测试
    # 將訓練好的生成器模型導出為獨立的 meta 文件，方便後續用於生成圖片而不依賴完整的訓練代碼。
    if a.mode == "export":
    # 如果 lab_colorization 設為 True，則拋出異常，因為該功能不支援導出。
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        # 定義輸入占位符，用於接收 Base64 格式的字符串數據
        input = tf.placeholder(tf.string, shape=[1])

        # 解析 Base64 字符串為原始影像數據
        input_data = tf.decode_base64(input[0])

        # 解碼 PNG 影像數據
        input_image = tf.image.decode_png(input_data)

        # 如果影像有 4 個通道 (RGBA)，則移除 Alpha 通道 (僅保留 RGB)
        input_image = tf.cond(
            tf.equal(tf.shape(input_image)[2], 4),
            lambda: input_image[:, :, :1],
            lambda: input_image
        )

        # 轉換影像數據類型為 float32，並將像素值歸一化到 [0, 1]
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)

        # 設定影像形狀：應為 [CROP_SIZE, 3 * CROP_SIZE, 1]
        input_image.set_shape([CROP_SIZE, 3 * CROP_SIZE, 1])

        # 增加批次維度 (batch dimension)，變為 [1, height, width, channels]
        batch_input = tf.expand_dims(input_image, axis=0)

        # 使用生成器處理影像
        with tf.variable_scope("generator"):
            batch_output = deprocess(
                create_generator(
                    preprocess(batch_input[:, :, :256, :]),  # 第一張
                    preprocess(batch_input[:, :, 256:512, :]),  # 第二張 (條件1)
                    preprocess(batch_input[:, :, 512:, :]),  # 第三張 (條件2)
                    1
                )
            )

        # 將生成的影像數據轉換為 uint8 (0-255)
        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]

        # 根據輸出格式選擇編碼方式 (PNG 或 JPEG)
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")

        # 轉換為 Base64 字符串格式，以便輸出
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        # 定義 key 的占位符，與輸入數據一起作為模型的輸入
        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))

        # 定義輸出的格式，將處理後的影像結果作為輸出
        outputs = {
            "key": tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        # 初始化模型變量
        init_op = tf.global_variables_initializer()

        # 設定 Saver 物件以恢復與保存模型權重
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            # 執行變量初始化
            sess.run(init_op)

            print("loading model from checkpoint")
            # 載入最近的 checkpoint
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)

            print("exporting model")
            # 將模型的計算圖 (meta graph) 保存為 .meta 文件
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))

            # 保存模型的權重
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    # 从数据集加载样本，返回的数据通常包含输入、目标图像以及相关的路径信息。
    examples = load_examples()
    print("examples count = %d" % examples.count)
    # create_model：创建生成对抗网络（GAN）模型。
    # 输入：examples.inputs 是模型的输入图像，examples.condition1/condition2 是条件信息，examples.targets 是目标图像。
    # 输出：包含生成的图像 (model.outputs)、真假预测 (model.predict_real 和 model.predict_fake) 和各种损失。
    model = create_model(examples.inputs, examples.condition1, examples.condition2, examples.targets)

    # 如果 lab_colorization 为 True，则输入和目标图像需要特殊处理：
    # AtoB 模式：
    # targets 和 outputs 增强：将输入的亮度信息加到目标和输出图像上。
    # inputs 去处理：将输入图像去处理为单通道灰度图像。
    # BtoA 模式：
    # inputs 增强：将目标图像的亮度信息加到输入图像上。
    # targets 和 outputs 去处理：去处理为正常 RGB 图像。
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    # 如果 lab_colorization 为 False，则直接对输入、目标和生成图像应用去处理：
    # cenSulFake 和 cenSulReal：模型对于真实和生成图像的真假预测结果。
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)
        cenSulFake = deprocess(model.predict_fake)
        cenSulReal = deprocess(model.predict_real)
    # 宽高比调整：如果 a.aspect_ratio 不为 1，则按比例调整图像的宽高。
    # 类型转换：将图像数据转换为整型（uint8），便于编码和显示。
    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    def save_images_hide(name,fetches):
        if not os.path.exists(name):
            os.makedirs(name)
        inputs_save = deprocess(fetches)
        shape_list=inputs_save.shape
        len_list=len(shape_list)
        num_batch=shape_list[len_list-1]
        #print(num_batch)
        #print(shape_list)
        for i in range(num_batch):
            image=inputs_save[0,:,:,i]
            cv2.imwrite(name+ str(i) + '.jpg', image)


        #im = Image.fromarray(image)
        #im.save('/home/yuanfly/pix2pix/facades/RestoreTeeth/img.jpg')
        #converted_saves = convert(inputs_save)

        #matplotlib.image.imsave('/home/yuanfly/pix2pix/facades/RestoreTeeth/img.png', inputs_save)
        #with open('/home/yuanfly/pix2pix/facades/RestoreTeeth/img.txt', "wb") as f:
            #f.write(inputs_save)
        return
    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("convert_cenSulFake"):
        converted_cenSulFake = convert(cenSulFake)

    with tf.name_scope("convert_cenSulReal"):
        converted_cenSulReal = convert(cenSulReal)
    # 对功能：将转换后的图像编码为 PNG 格式，并存储到 display_fetches 中，以便后续保存或显示。
    # 路径信息：examples.paths 保存了与每个图像对应的文件路径。
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
    # 将输入、目标、生成的图像以及真假预测的结果写入 TensorBoard，便于可视化模型的训练效果。
    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("cenSulFake_summary"):
        tf.summary.image("sulFake", converted_cenSulFake)

    with tf.name_scope("cenSulReal_summary"):
        tf.summary.image("sulReal", converted_cenSulReal)
    #with tf.name_scope("predict_real_summary"):
        #tf.summary.image("predict_local_real0", tf.image.convert_image_dtype(model.predict_local_real0, dtype=tf.uint8))
        #tf.summary.image("predict_local_real1", tf.image.convert_image_dtype(model.predict_local_real1, dtype=tf.uint8))
        #tf.summary.image("predict_local_real2", tf.image.convert_image_dtype(model.predict_local_real2, dtype=tf.uint8))

    #with tf.name_scope("predict_fake_summary"):
        #tf.summary.image("predict_local_fake0", tf.image.convert_image_dtype(model.predict_local_fake0, dtype=tf.uint8))
        #tf.summary.image("predict_local_fake1", tf.image.convert_image_dtype(model.predict_local_fake1, dtype=tf.uint8))
        #tf.summary.image("predict_local_fake2", tf.image.convert_image_dtype(model.predict_local_fake2, dtype=tf.uint8))

    # with tf.name_scope("predict_real_summary"):
    #     tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    #
    # with tf.name_scope("predict_fake_summary"):
    #     tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
    # 记录 GAN 训练中的各种损失：
    # 判别器损失：
    # global_discriminator_loss：全局判别器损失。
    # local_discriminator_loss：局部判别器损失。
    # discriminator_loss_per：与感知相关的判别器损失。
    # 生成器损失：
    # gen_loss_GAN：生成器对抗损失。
    # gen_loss_L1：生成器 L1 损失（图像重建）。
    # gen_loss_CenSul：生成器中心沟损失（特定任务相关）。
    # gen_per_loss：感知损失。
    # hist_loss：直方图损失（衡量分布差异）。
    tf.summary.scalar("generator_loss_collision", model.gen_loss_collision)
    tf.summary.scalar("global_discriminator_loss", model.global_discrim_loss)
    tf.summary.scalar("local_discriminator_loss", model.local_discrim_loss)
    tf.summary.scalar("discriminator_loss_per", model.discrim_loss_per)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_cenSul", model.gen_loss_CenSul)
    tf.summary.scalar("generator_loss_per", model.gen_per_loss)
    tf.summary.scalar("histogram_loss", model.histogram_loss)

    # 對所有可訓練的變數（如權重、偏置）繪製直方圖，記錄其值的分布
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    # 記錄生成器與判別器中變數的梯度分布。
    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)
    # 功能：計算所有可訓練參數的總數量，通常用於估計模型的複雜度。
    # tf.reduce_prod：計算每個變數的元素數量（即張量維度的積）。
    # tf.reduce_sum：將所有參數的數量加總。
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])


    # saver = tf.train.Saver(max_to_keep=1)
    #fly modify
    #     功能：獲取 generator 範疇中的所有可訓練變數，並為其建立一個 Saver 物件。
    # 目的：從檢查點中恢復生成器的權重與偏置，但這些參數不參與之後的反向傳播。
    #获取牙齿修复模型（牙齿沟窝提取）中生成器G的G的卷积核，接下来恢复卷积核的权重和偏置 并且两者不参与反向传播 （word博客上有介绍）
    ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    saver = tf.train.Saver(ref_vars)
    # 功能：同樣針對 genTeethGroove 範疇的變數進行保存和恢復操作。
    # 目的：用於專門處理牙齒溝槽的生成模型，權重與偏置被獨立管理。
    cenSul_ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='genTeethGroove')
    cenSulsaver = tf.train.Saver(cenSul_ref_vars)
    # logdir：指定保存日誌的目錄，根據 trace_freq 和 summary_freq 判斷是否啟用。
    # Supervisor：TensorFlow 的高級 API，用於管理會話、日誌保存以及檢查點。

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    # 功能：啟動受 Supervisor 管理的會話。
    # 參數計算：列印總參數數量，便於檢查模型規模
    with sv.managed_session() as sess:
        # print parameter——count
    # 功能：從指定路徑加載最新的檢查點，恢復生成器的權重與偏置。
    # 目的：在接續訓練或進行測試時，避免從頭開始訓練。
        print("parameter_count =", sess.run(parameter_count))
        if a.checkpoint is not None:
            print("loading teeth repair model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
        # 針對中央溝槽模型，從 a.cktCentralSul 指定的路徑加載權重。
        if a.cktCentralSul is not None:
            print("loading central groove model from checkpoint")
            ckpt = tf.train.get_checkpoint_state(a.cktCentralSul)
            cenSulsaver.restore(sess, ckpt.model_checkpoint_path)
        # fly modify
        # a.max_epochs：若指定最大訓練世代，步數為每個世代的步數乘以總世代數。
        # a.max_steps：直接指定總步數，覆蓋之前的計算。
        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps
        # 限制測試步數為測試數據的總步數與設定的最大步數（max_steps）之間的最小值。
        # 在每一步：
        # 執行測試：通過 sess.run 執行測試過程，生成圖片結果（results）。
        # 保存測試結果：利用 save_images(results) 保存輸出的圖片。
        # 記錄測試圖片名稱：在終端輸出測試圖片的名稱。
        # 更新索引文件：通過 append_index 添加生成圖片的索引，便於後續檢
        # 测试的入口
        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("测试 image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()
            # 訓練循環運行至 max_steps 設定的最大步數。
            # 每一步，根據不同頻率（freq）執行對應的操作。
            for step in range(max_steps):
            # 用途：判斷是否需要執行某些操作（如保存模型、記錄摘要等）。
            # 條件：當頻率大於 0 且當前步數符合頻率條件，或者是最後一步。
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = tf.RunMetadata()
                #                 if should(a.trace_freq):
                #                     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #                     run_metadata = tf.RunMetadata()
                # model.train：執行單次訓練步驟。
                # sv.global_step：取得全域步數。
                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }
                # 根據頻率添加操作：
                # 記錄損失值（如 global_discrim_loss, gen_loss_L1）。
                # 保存摘要（summary）。
                # 保存顯示圖片（display）。
                # 記錄執行痕跡（trace）。

                if should(a.progress_freq):
                    fetches["global_discrim_loss"] = model.global_discrim_loss
                    fetches["local_discrim_loss"] = model.local_discrim_loss
                    fetches["discrim_loss_per"] = model.discrim_loss_per
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_CenSul"] = model.gen_loss_CenSul
                    fetches["gen_per_loss"] = model.gen_per_loss
                    fetches["histogram_loss"] = model.histogram_loss
                    fetches["gen_loss_collision"] = model.gen_loss_collision  # 新增

                # 用 sess.run 執行定義的操作，並返回結果 results。
                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op
                #  不同頻率執行的操作
                if should(a.display_freq):
                    fetches["display"] = display_fetches
                # if should(a.saveHide_freq):
                #     fetches["hide_layer3"] =model.predict_local_fake2
                #     fetches["hide_layer2"] = model.predict_local_fake1
                #     fetches["hide_layer1"] = model.predict_local_fake0

                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                # 每隔 summary_freq 步記錄一次 TensorBoard 摘要，便於後續分析。
                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                # 保存生成的顯示圖片，並更新索引文件。
                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)
                # 使用 RunMetadata 記錄完整的運行痕跡，供 TensorBoard 分析模型性能（如運算圖、內存占用）。
                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])
                # 顯示信息：
                # 當前訓練的世代（epoch）與步數。
                # 處理速率（圖片/秒）。
                # 預計剩餘時間（分鐘）。
                # 列印損失值：顯示生成器和判別器的各種損失。
                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    print("global_discrim_loss", results["global_discrim_loss"])
                    print("local_discrim_loss", results["local_discrim_loss"])
                    print("discrim_loss_per", results["discrim_loss_per"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    print("gen_loss_CenSul", results["gen_loss_CenSul"])
                    print("gen_per_loss", results["gen_per_loss"])
                    print("histogram_loss", results["histogram_loss"])
                    print("gen_loss_collision", results["gen_loss_collision"])  # 新增
                    
                    

                # 每隔 save_freq 步保存一次模型的檢查點。
                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                # 使用 Supervisor 判斷是否需要中斷訓練。
                # if should(a.saveHide_freq):
                #     print("保存中间层图像")
                #     layer='D://Users//user//Desktop//weiyundontdelete//GANdata//trainingdepth//DAISdepth//alldata//layer//'
                #     save_images_hide(layer+str(3)+'/'+str(results["global_step"])+'/',results["hide_layer3"])
                #     save_images_hide(layer+str(2)+'/'+ str(results["global_step"])+'/',results["hide_layer2"])
                #     save_images_hide(layer+str(1)+'/'+ str(results["global_step"])+'/',results["hide_layer1"])
                
                if sv.should_stop():
                    break


main()
