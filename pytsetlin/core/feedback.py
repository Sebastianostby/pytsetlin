from numba import njit, prange
import numpy as np
from pytsetlin.core import config


@njit(parallel=config.OPERATE_PARALLEL)
def evaluate_clauses_training(literals, cb, n_literals, clause_outputs, literals_counts):

    # captures the imply operation ta -> lit?

    clause_outputs.fill(1)

    literals_counts.fill(0)
    
    for clause_k in prange(cb.shape[0]):

        pos_literal_count = 0
        neg_literal_count = 0

        for literal_k in range(n_literals):

            if(cb[clause_k, literal_k] > 0):


                if(literals[literal_k] == 0):

                    clause_outputs[clause_k] = 0

                    break
            
                pos_literal_count += 1

            if(cb[clause_k, literal_k + n_literals] > 0):


                if(literals[literal_k] == 1):   

                    clause_outputs[clause_k] = 0

                    break

                neg_literal_count += 1

        literals_counts[clause_k] = pos_literal_count + neg_literal_count



@njit
def evaluate_clause(literals, clause_block, n_literals):

    clause_outputs = np.ones(clause_block.shape[0], dtype=np.uint8)

    for clause_k in range(clause_block.shape[0]):

        is_empty_clause = True

        for literal_k in range(n_literals):

            if(clause_block[clause_k, literal_k] > 0):

                is_empty_clause = False

                if(literals[literal_k] == 0):

                    clause_outputs[clause_k] = 0

                    break
            
            if(clause_block[clause_k, literal_k + n_literals] > 0):

                is_empty_clause = False

                if(literals[literal_k] == 1):    

                    clause_outputs[clause_k] = 0

                    break

        if(is_empty_clause):

            clause_outputs[clause_k] = 0


    return clause_outputs


@njit
def get_update_p(vote_values, threshold, y, target_class):
  
    vote_value = np.clip(np.array(vote_values[y]), -threshold, threshold)

    if target_class:
        return (threshold - vote_value) / (2 * threshold)
    else:
        return (threshold + vote_value) / (2 * threshold)



@njit(parallel=config.OPERATE_PARALLEL)
def update_clauses(cb, wb, clause_outputs, literals_counts, positive_prob, negative_prob, target, not_target, literals, n_literals, n_literal_budget, s_min_inv, s_inv, boost_true_positives):
    
    for clause_k in prange(cb.shape[0]):

        if literals_counts[clause_k] > n_literal_budget:
            clause_outputs[clause_k] = 0

        if np.random.random() <= positive_prob:
            update_clause(cb, wb, 1, literals, n_literals, clause_outputs, clause_k, target, s_min_inv, s_inv, boost_true_positives) 

        if np.random.random() <= negative_prob:
            update_clause(cb, wb, -1, literals, n_literals, clause_outputs, clause_k, not_target, s_min_inv, s_inv, boost_true_positives) 
 



@njit
def update_clause(cb, wb, target, literals, n_literals, clause_output, clause_k, class_k, s_min_inv, s_inv, boost_true_positives):
    
    sign = 1 if (wb[class_k, clause_k] >= 0) else -1

    if(target*sign > 0):

        if clause_output[clause_k] == 1:

            wb[class_k, clause_k] += sign

            T1aFeedback(cb, clause_k, literals, n_literals, s_min_inv, s_inv, boost_true_positives)

        else:
            
            T1bFeedback(cb, clause_k, n_literals, s_inv)

    elif(target*sign < 0):

        if clause_output[clause_k] == 1:

            wb[class_k, clause_k] -= sign

            T2Feedback(cb, clause_k, literals, n_literals)


@njit
def T1aFeedback(cb, clause_k, literals, n_literals, s_min_inv, s_inv, boost_true_positives):
    
    upper_state =  127
    lower_state = -127

    for literal_k in range(n_literals):

        if(literals[literal_k] == 1):

            # here we boost true possitives
            if boost_true_positives:
                if(cb[clause_k, literal_k] < upper_state):
                    cb[clause_k, literal_k] += 1

            else:
                if(np.random.random() <= s_min_inv):
                    if(cb[clause_k, literal_k] < upper_state):
                        cb[clause_k, literal_k] += 1

            if(np.random.random() <= s_inv):
                if(cb[clause_k, literal_k + n_literals] > lower_state):
                    cb[clause_k, literal_k + n_literals] -= 1          
            

        else:
            if(np.random.random() <= s_inv):
                if(cb[clause_k, literal_k] > lower_state):
                    cb[clause_k, literal_k] -= 1

            # here we boost true possitives
            if boost_true_positives:
                if(cb[clause_k, literal_k + n_literals] < upper_state):
                    cb[clause_k, literal_k + n_literals] += 1       

            else:
                if(np.random.random() <= s_min_inv):
                    if(cb[clause_k, literal_k + n_literals] < upper_state):
                        cb[clause_k, literal_k + n_literals] += 1                           

@njit
def T1bFeedback(cb, clause_k, n_literals, s_inv):

    lower_state = -127
    
    for literal_k in range(n_literals):

        if np.random.random() <= s_inv:

            if cb[clause_k, literal_k] > lower_state:
                cb[clause_k, literal_k] -= 1

        if np.random.random() <= s_inv:

            if cb[clause_k, literal_k + n_literals] > lower_state:
                cb[clause_k, literal_k + n_literals] -= 1

@njit
def T2Feedback(cb, clause_k, literals, n_literals):

    for literal_k in range(n_literals):

        if(literals[literal_k] == 0):
            if(cb[clause_k, literal_k] <= 0):
                cb[clause_k, literal_k] += 1

        else:
            if(cb[clause_k, literal_k + n_literals] <= 0):
                
                cb[clause_k, literal_k + n_literals] += 1