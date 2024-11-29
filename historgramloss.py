import os
# 選擇 CUDA 版本
cuda_version = "v8.0"  # 修改為 "v11.7" 或其他版本

# 動態設置環境變數
os.environ["CUDA_HOME"] = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}"
os.environ["Path"] = f"{os.environ['CUDA_HOME']}\\bin;{os.environ['CUDA_HOME']}\\libnvvp;" + os.environ["Path"]
import tensorflow as tf

#import tensorflow as tf
# 計算特徵向量之間的餘弦相似度。若未提供 Y，則使用 X 自身。
# 回傳一個形狀為 [n, n] 的矩陣，其中每個元素表示樣本間的餘弦相似度。
def cos(X, Y=None):
    X_n = tf.math.l2_normalize(X, axis=1)  # L2 正規化
    if (Y is None) or (X is Y):
        return tf.matmul(X_n, tf.transpose(X_n))  # 計算餘弦相似度
    Y_n = tf.math.l2_normalize(Y, axis=1)
    return tf.matmul(X_n, tf.transpose(Y_n))


# 根據標籤構造相似性矩陣。
# 回傳一個形狀為 [n, n] 的二值矩陣，其中元素為 1 表示樣本對共享至少一個相同標籤。
def sim_mat(label, label2=None):
    """S[i][j] = 1 <=> i- & j-th share at lease 1 label"""
    if label2 is None:
        label2 = label
    return tf.cast(tf.matmul(label, tf.transpose(label2)) > 0, "float32")

# X (特徵向量):

# 形狀 [n, d]，其中 n 是樣本數量，d 是特徵的維度。
# 它是模型輸出的特徵向量，未經 L2 標準化。
# L (標籤向量):

# 形狀 [n, c]，其中 c 是類別數量。
# 每一行表示該樣本的標籤，通常為 one-hot 編碼或多標籤格式。
# R (直方圖估計點的數量):

# 一個標量，用來控制質方圖的分辨率（估計點的數量）。默認值為 151。
def histogram_loss(X, L, R=151):
    """histogram loss
    X: [n, d], feature WITHOUT L2 norm
    L: [n, c], label
    R: scalar, num of estimating point, same as the paper
    """
    delta = 2. / (R - 1)  # step
    # t = (t_1, ..., t_R)
    t = tf.lin_space(-1., 1., R)[:, None]  # [R, 1]
    # gound-truth similarity matrix
    M = sim_mat(L)  # [n, n]
    # cosine similarity, in [-1, 1]
    S = cos(X)  # [n, n]

    # get indices of upper triangular (without diag)
    S_hat = S + 2  # shift value to [1, 3] to ensure triu > 0
    S_triu = tf.linalg.band_part(S_hat, 0, -1) * (1 - tf.eye(tf.shape(S)[0]))
    triu_id = tf.where(S_triu > 0)

    # extract triu -> vector of [n(n - 1) / 2]
    S = tf.gather_nd(S, triu_id)[None, :]  # [1, n(n-1)/2]
    M_pos = tf.gather_nd(M, triu_id)[None, :]
    M_neg = 1 - M_pos

    scaled_abs_diff = tf.math.abs(S - t) / delta  # [R, n(n-1)/2]
    # mask_near = tf.cast(scaled_abs_diff <= 1, "float32")
    # delta_ijr = (1 - scaled_abs_diff) * mask_near
    delta_ijr = tf.maximum(0, 1 - scaled_abs_diff)

    def histogram(mask):
        """h = (h_1, ..., h_R)"""
        sum_delta = tf.reduce_sum(delta_ijr * mask, 1)  # [R]
        return sum_delta / tf.maximum(1, tf.reduce_sum(mask))

    h_pos = histogram(M_pos)[None, :]  # [1, R]
    h_neg = histogram(M_neg)  # [R]
    # all 1 in lower triangular (with diag)
    mask_cdf = tf.linalg.band_part(tf.ones([R, R]), -1, 0)
    cdf_pos = tf.reduce_sum(mask_cdf * h_pos, 1)  # [R]

    loss = tf.reduce_sum(h_neg * cdf_pos)
    return loss
