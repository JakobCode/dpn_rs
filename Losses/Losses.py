import tensorflow as tf
from Measures.Measures import dirichlet_kl_divergence, dirichlet_kl_divergence_reverse, concentrations_from_logits
from Callbacks.ENN_lambda_update import ENN_lambda_update
from settings import Approaches

def get_loss_function(approach: Approaches):
    if approach is Approaches.dpn_rs:
        return dpn_rs, []
    elif approach is Approaches.prior_kl_forward:
        return prior_forward, []
    elif approach is Approaches.prior_kl_reverse:
        return prior_reverse, []
    elif approach is Approaches.dpn_plus:
        return dpn_plus, []
    elif approach is Approaches.enn_cross_entropy:
        cb = ENN_lambda_update()
        return build_evidential_cross_entropy(cb), [cb]
    elif approach is Approaches.baseline:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True), []

def dpn_rs(y_true, logits):
    logits = tf.clip_by_value(logits, -20, 20)
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=logits, from_logits=True)
    lambda_p = (tf.reduce_max(y_true, axis=-1) - 0.5)
    y_sgm = tf.nn.sigmoid(logits)
    regularizer = lambda_p * tf.reduce_mean(y_sgm, axis=-1)

    return loss - regularizer

def prior_forward(alpha_target, logits):
    alpha_pred = concentrations_from_logits(logits)
    alpha_target = tf.cast(alpha_target, tf.float64)
    return dirichlet_kl_divergence(alpha_target, alpha_pred)

def prior_reverse(alpha_target, logits):
    alpha_pred = concentrations_from_logits(logits)
    alpha_target = tf.cast(alpha_target, tf.float64)
    return dirichlet_kl_divergence_reverse(alpha_target, alpha_pred)

def dpn_plus(y_true, logits):
    logits = tf.clip_by_value(logits, -20, 20)
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true,y_pred=logits, from_logits=True)
    lambda_p = (tf.reduce_max(y_true, axis=-1) + 0.5)
    y_sgm = tf.nn.sigmoid(logits)
    regularizer = lambda_p * tf.reduce_mean(y_sgm, axis=-1)

    return loss - regularizer

def build_evidential_cross_entropy(lambda_callback : ENN_lambda_update):
    def evidential_cross_entropy(y_true, logits):
        y_true = tf.cast(y_true, tf.float64)

        alpha_c = concentrations_from_logits(logits) + 1
        S = tf.reduce_sum(alpha_c, axis=1, keepdims=True)
        E = alpha_c - 1

        A = tf.reduce_sum(y_true * (tf.math.digamma(S) - tf.math.digamma(alpha_c)), axis=1)

        annealing_coef = lambda_callback.lambda_t

        alp = E * (1 - y_true) + 1
        beta = tf.ones_like(alp)

        B = annealing_coef * dirichlet_kl_divergence(alp, beta)

        return A + B

    return evidential_cross_entropy


def KL(alpha_c):
    beta = tf.ones_like(alpha_c)
    S_alpha = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=-1, keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha_c), axis=-1, keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=-1, keepdims=True) - tf.math.lgamma(S_beta)

    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha_c)

    kl = tf.squeeze(tf.reduce_sum((alpha_c - beta) * (dg1 - dg0), axis=-1, keepdims=True) + lnB + lnB_uni)
    return kl