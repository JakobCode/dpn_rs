from scipy.special import psi, gammaln
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def dirichlet_kl_divergence(alpha_c_target, alpha_c_pred, eps=10e-10):

    alpha_0_target = tf.reduce_sum(alpha_c_target, axis=-1, keepdims=True)
    alpha_0_pred = tf.reduce_sum(alpha_c_pred, axis=-1, keepdims=True)

    term1 = tf.math.lgamma(alpha_0_target) - tf.math.lgamma(alpha_0_pred)
    term2 = tf.math.lgamma(alpha_c_pred + eps) - tf.math.lgamma(alpha_c_target + eps)

    term3_tmp = tf.math.digamma(alpha_c_target + eps) - tf.math.digamma(alpha_0_target + eps)
    term3 = (alpha_c_target - alpha_c_pred) * term3_tmp

    result = tf.squeeze(term1 + tf.reduce_sum(term2 + term3, keepdims=True, axis=-1))

    return result


def dirichlet_kl_divergence_reverse(alpha_c_target, alpha_c_pred, eps=10e-10):
    return dirichlet_kl_divergence(alpha_c_target=alpha_c_pred,
                                   alpha_c_pred=alpha_c_target,
                                   eps=eps)

def concentrations_from_logits(logits):
    logits = tf.cast(logits, tf.float64)
    alpha_c = tf.exp(logits)
    alpha_c = tf.clip_by_value(alpha_c, clip_value_min=10e-25, clip_value_max=10e25)
    return alpha_c

def log_sum(logits, keepdims=False):
    return tf.reduce_sum(logits, axis=-1, keepdims=keepdims)

def probability_from_concentration(alpha_c):
    alpha_0 = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)
    return alpha_c / alpha_0


def probability_from_logits(logits):
    alpha_c = concentrations_from_logits(logits)
    prob = probability_from_concentration(alpha_c)
    return prob


def max_probability_from_logits(logits):
    prob = probability_from_logits(logits)
    return tf.reduce_max(prob, axis=-1)


def precision_from_logits(logits, keepdims=False):
    alpha_c = concentrations_from_logits(logits)
    return tf.reduce_sum(alpha_c, axis=-1, keepdims=keepdims)


def entropy(logits, keepdims=False):
    prob = probability_from_logits(logits)
    return -tf.reduce_sum(prob * tf.math.log(prob), axis=-1, keepdims=keepdims)


def differential_entropy(logits):
    alpha_c = concentrations_from_logits(logits)
    alpha_0 = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)

    lgamma_alpha_c = tf.math.lgamma(alpha_c)
    lgammaln_alpha_0 = tf.math.lgamma(alpha_0)

    digamma_alpha_c = tf.math.digamma(alpha_c)
    digamma_alpha_0 = tf.math.digamma(alpha_0)

    temp_mat = tf.reduce_sum((alpha_c - 1) * (digamma_alpha_c - digamma_alpha_0), axis=-1)
    metric = tf.reduce_sum(lgamma_alpha_c, axis=-1) - lgammaln_alpha_0 - temp_mat
    return metric


def mutual_information(logits):
    alpha_c = concentrations_from_logits(logits)

    alpha_0 = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)

    digamma_alpha_c = tf.math.digamma(alpha_c + 1)
    digamma_alpha_0 = tf.math.digamma(alpha_0 + 1)
    alpha_div = alpha_c / alpha_0

    temp_mat = tf.reduce_sum(- alpha_div * (tf.math.log(alpha_c) - digamma_alpha_c), axis=-1)
    metric = temp_mat + tf.squeeze(tf.math.log(alpha_0) - digamma_alpha_0)

    return metric

def differential_entropy (logits):
    logits = logits.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=-1)
    gammaln_alpha_c = gammaln(alpha_c)
    gammaln_alpha_0 = gammaln(alpha_0)
    
    psi_alpha_c = psi(alpha_c)
    psi_alpha_0 = psi(alpha_0)
    psi_alpha_0 = np.expand_dims(psi_alpha_0, axis = 1)
    
    temp_mat = np.sum((alpha_c-1)*(psi_alpha_c-psi_alpha_0), axis = 1)
    
    metric = np.sum(gammaln_alpha_c, axis=-1) - gammaln_alpha_0 - temp_mat
    return metric
    
def mutual_info(logits):
    logits = logits.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=-1, keepdims = True)
 
    psi_alpha_c = psi(alpha_c+1)
    psi_alpha_0 = psi(alpha_0+1)
    alpha_div = alpha_c / alpha_0
    
    temp_mat = np.sum(- alpha_div*(np.log(alpha_c) - psi_alpha_c), axis=-1)
    metric = temp_mat + np.squeeze(np.log(alpha_0) - psi_alpha_0)
    return metric

def _get_prob(logits):
    logits = logits.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-25, 10e25)
    alpha_0 = np.sum(alpha_c, axis=-1)
    alpha_0 = np.expand_dims(alpha_0, axis=-1)
    
    return (alpha_c/ alpha_0)
    
def entropy(logits):
    logits = logits.astype('float64')
    prob = np.clip(_get_prob(logits), a_min=10e-25, a_max=np.inf)
    exp_prob = np.log(prob)

    ent = -np.sum(prob*exp_prob, axis=-1)
    return ent

def max_prob (logits):
    logits = logits.astype('float64')
    prob = _get_prob(logits)
    metric = np.max(prob, axis=-1)
    return metric

def get_scores(in_predict, out_predict, save_path="./", name="", evidential=False):

    name = name + "_" if name != "" else ""
    gt_neg_in = np.zeros_like(in_predict[:,0])
    gt_neg_out = np.ones_like(out_predict[:,0])
    gtNeg = np.append(gt_neg_in, gt_neg_out, axis= 0)

    logits = np.append(in_predict, out_predict, axis= 0)

    roc_prob = round(roc_auc_score(gtNeg, -max_prob(logits)) *100, 2)
    roc_log_sum = round(roc_auc_score(gtNeg, -log_sum(logits)) *100, 2)
    roc_precision = round(roc_auc_score(gtNeg, -precision_from_logits(logits))*100, 2)
    roc_mi = round(roc_auc_score(gtNeg, mutual_info(logits)) *100, 2)
    roc_ent = round(roc_auc_score(gtNeg, entropy(logits)) *100, 2)

    print(f"Seperation Performance AUROC scores: \n",
          f"     max. probability: {roc_prob}\n",
          f"              Log-Sum: {roc_log_sum}\n",
          f"            Precision: {roc_precision}\n",
          f"   Mutual Information: {roc_mi}\n",
          f"              Entropy: {roc_ent}\n")