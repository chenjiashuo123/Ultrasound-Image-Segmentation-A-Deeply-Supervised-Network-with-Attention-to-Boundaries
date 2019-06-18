import tensorflow as tf


def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)#将label格式进行转换

    count_neg = tf.reduce_sum(1. - y)#降维，计算0的个数
    count_pos = tf.reduce_sum(y)#计算1的个数

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    # 对于加了权值pos_weight的交叉熵函数:targets∗−log(sigmoid(logits))∗pos_weight+(1−targets)∗−log(1−sigmoid(logits))
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
    #a,b为和ｔｅｎｓｏｒ相同维度的ｔｅｎｓｏｒ，将ｔｅｎｓｏｒ中的ｔｒｕｅ位置元素替换为ａ中对应位置元素，ｆａｌｓｅ的替换为ｂ中对应位置元素
