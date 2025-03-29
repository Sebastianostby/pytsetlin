from numba import njit, prange
import numpy as np
from pytsetlin.core.feedback import evaluate_clauses_training, update_clauses, evaluate_clause
from pytsetlin.core import config


@njit
def train_epoch(cb, wb, x, y, threshold, s_min_inv, s_inv, n_outputs, n_literals, n_literal_budget, boost_true_positives, seed):
    
    np.random.seed(seed)

    n_indices = len(x)

    clause_outputs = np.ones(cb.shape[0], dtype=np.uint8)
    literals_counts = np.zeros(cb.shape[0], dtype=np.uint32)

    y_hat = np.zeros(n_indices, dtype=np.uint32)

    for indice in range(n_indices):
        literals = x[indice]
        target = y[indice]
        
        evaluate_clauses_training(literals, cb, n_literals, clause_outputs, literals_counts)
        
        vote_values = np.dot(wb.astype(np.float32), clause_outputs.astype(np.float32))

        vote_values_clamped = np.clip(vote_values,  -threshold, threshold)

        not_target = np.random.randint(0, n_outputs)

        while(not_target == target):
            not_target = np.random.randint(0, n_outputs)

        v_clamped_pos = vote_values_clamped[target]

        pos_update_p =  (threshold - v_clamped_pos) / (2*threshold)

        v_clamped_neg = vote_values_clamped[not_target]

        neg_update_p =  (threshold + v_clamped_neg) / (2*threshold)

        update_clauses(cb, wb, clause_outputs, literals_counts, pos_update_p, neg_update_p, target, 
                    not_target, literals, n_literals, n_literal_budget, s_min_inv, s_inv, boost_true_positives)

        y_hat[indice] = np.argmax(vote_values)

    return y_hat


@njit
def classify(x, clause_block, weight_block, n_literals):

    clause_outputs = evaluate_clause(x, clause_block, n_literals)

    class_sums = np.dot(weight_block.astype(np.float32), clause_outputs.astype(np.float32))

    return np.argmax(class_sums)
  
@njit(parallel=config.OPERATE_PARALLEL)
def eval_predict(x, cb, wb, n_literals):
    y_pred = np.zeros(x.shape[0], dtype=np.uint32)
    
    for i in prange(x.shape[0]):
        y_pred[i] = classify(x[i], cb, wb, n_literals)
    
    return y_pred
