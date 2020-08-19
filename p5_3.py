from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from conlleval import evaluate
from scipy.optimize import fmin_l_bfgs_b
import json


def read_data(path, column=0):
    """column=0 means input sequence, column=1 means label
    """
    with open(path) as f:
        lines = f.readlines()
    
    data = []
    sample = []
    
    for line in lines:
        formatted_line = line.strip()
        
        if len(formatted_line) > 0:
            split_data = formatted_line.split(" ")
            sample.append(split_data[column])

        else:
            data.append(sample)
            sample = []
            
    return data


def calc_e(x_data, y_data, x_vocab, y_vocab):
    count_emission = Counter([(x,y) for x_instance, y_instance in zip(x_data, y_data) for x, y in zip(x_instance, y_instance)])
    count_label = Counter([oo for o in y_data for oo in o])
    
    e_score = {}
    for y in y_vocab:
        for x in x_vocab:
            feature = f"emission:{y}+{x}"
            
            if (x,y) not in count_emission:
                e_score[feature] = ninf
            else:
                score = np.log(count_emission[(x,y)]  /  count_label[y])
                e_score[feature] = score
    
    return e_score

def calc_t(y_data, y_vocab):
    count_transition = Counter([ (y_prev, y) for y_instance in y_data for y_prev, y in zip(['START'] + y_instance, y_instance + ['STOP'])])
    count_label = Counter([y for y_instance in y_data for y in ['START'] + y_instance])
    
    f_score = {}
    for y_prev in ['START'] + y_vocab:
        for y in y_vocab + ['STOP']:
            feature = f"transition:{y_prev}+{y}"
            
            if (y_prev,y) not in count_transition:
                f_score[feature] = ninf
            else:
                score = np.log(count_transition[(y_prev,y)]  /  count_label[y_prev])
                f_score[feature] = score
    
    return f_score

def calc_new(x_data, y_data, y_vocab):
    count_numerator = []
    for x_instance, y_instance in zip(x_data, y_data):
        for x, y_prev, y in zip(x_instance, ['START'] + y_instance[:-1], y_instance):
            count_numerator.append( (y_prev, y, x) )
    count_numerator = Counter(count_numerator)
    count_denom = Counter([ y for y_instance in y_data for y in ['START'] + y_instance[:-1] ])
    
    f_score = {}
    for (y_prev,y,x), numerator in dict(count_numerator).items():
        feature = f"transition:{y_prev}+{y}+{x}"
        f_score[feature] = np.log( numerator  /  count_denom[y_prev])
    
    return f_score


def compute_score(x_instance, x_instance_pos, y_instance, feature_dict):
    feature_count = defaultdict(int)
    
    for x, y in zip(x_instance, y_instance): feature_count[f"emission:{y}+{x}"] += 1
    for x, y in zip(x_instance_pos, y_instance): feature_count[f"emission:{y}+{x}"] += 1

    for y_prev, y in zip(['START'] + y_instance, y_instance + ['STOP']):
        feature_count[f"transition:{y_prev}+{y}"] += 1
    for y_prev, y, x in zip( ['START']+ y_instance[:-1], y_instance, x_instance ):
        feature_count[f"transition:{y_prev}+{y}+{x}"] += 1
    score = 0
    for feat, count in feature_count.items():
        if feat in feature_dict:
            score += feature_dict[feat]*count
        else:
            score += ninf*count
    return score



def viterbi(x_instance, x_instance_pos, y_vocab, feature_dict):
    n, d = len(x_instance), len(y_vocab)
    scores = np.full( (n,d), -np.inf) # initialize to be very negative
    bp = np.full( (n,d), 0, dtype=np.int)
    
    for i, y in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:START+{y}",  ninf)
        e_score = feature_dict.get( f"emission:{y}+{x_instance[0]}",  ninf)
        e_score_pos = feature_dict.get(f"emission:{y}+{x_instance_pos[0]}", ninf)
        new_score = feature_dict.get(f"transition:START+{y}+{x_instance_pos[0]}", ninf)
        scores[0, i] = t_score + e_score + e_score_pos + new_score
        
    for i in range(1, n):
        for y_i, y in enumerate(y_vocab):
            for y_prev_i, y_prev in enumerate(y_vocab):
                t_score = feature_dict.get( f"transition:{y_prev}+{y}", ninf)
                e_score = feature_dict.get( f"emission:{y}+{x_instance[i]}", ninf)
                e_score_pos = feature_dict.get( f"emission:{y}+{x_instance_pos[i]}", ninf)
                new_score = feature_dict.get(f"transition:{y_prev}+{y}+{x_instance[i]}", ninf)
                score = t_score + e_score + e_score_pos + new_score + scores[i-1, y_prev_i]
                if score > scores[i, y_i]:
                    scores[i, y_i] = score
                    bp[i, y_i] = y_prev_i
    
    final_score, final_bp = ninf, 0
    for i, y_prev in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:{y_prev}+STOP", ninf)
        score = t_score + scores[n-1, i]
        if score > final_score: 
            final_score = score
            final_bp = i
    decoded_sequence = [ y_vocab[final_bp], ]
    for i in range(n-1, 0, -1):
        final_bp = bp[i, final_bp]
        decoded_sequence = [ y_vocab[final_bp] ] + decoded_sequence
        
    return decoded_sequence


def inference(in_file_path, y_vocab, feature_dict, out_file_path):
    x_data = read_data(in_file_path, column=0)
    x_data_pos = read_data(in_file_path, column=1)
    y_preds = []
    with open(out_file_path, 'w') as f:
        for x_instance, x_instance_pos in zip(x_data, x_data_pos):
            pred = viterbi(x_instance, x_instance_pos, y_vocab, feature_dict)
            y_preds.append(pred)
    return y_preds



def logsumexp(a):
    return np.max(a) 

def forward(x_instance, x_instance_pos, y_vocab, feature_dict):
    n, d = len(x_instance), len(y_vocab)
    scores = np.zeros( (n,d) )
    
    for i, y in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:START+{y}", ninf)
        new_score = feature_dict.get( f"transition:START+{y}+{x_instance[0]}", ninf)
        scores[0, i] = t_score + new_score
    
    for i in range(1, n):
        for y_i, y in enumerate(y_vocab):
            temp = []
            for y_prev_i, y_prev in enumerate(y_vocab):
                t_score = feature_dict.get( f"transition:{y_prev}+{y}", ninf)
                e_score = feature_dict.get( f"emission:{y_prev}+{x_instance[i-1]}", ninf)
                e_score_pos = feature_dict.get( f"emission:{y_prev}+{x_instance_pos[i-1]}", ninf)
                new_score = feature_dict.get( f"transition:{y_prev}+{y}+{x_instance[i]}", ninf)
                temp.append(e_score + e_score_pos + t_score + new_score + scores[i-1, y_prev_i])
            scores[i, y_i] = logsumexp(np.array(temp))
    
    temp = []
    for i, y_prev in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:{y_prev}+STOP", ninf)
        e_score = feature_dict.get( f"emission:{y_prev}+{x_instance[-1]}", ninf)
        e_score_pos = feature_dict.get( f"emission:{y_prev}+{x_instance_pos[-1]}", ninf)
        temp.append(e_score + e_score_pos + t_score + scores[-1, i])
    alpha = logsumexp(np.array(temp))
    
    return scores, alpha



def loss_fn_instance(x_instance, x_instance_pos, y_instance, feature_dict, y_vocab):
    first_term = compute_score(x_instance, x_instance_pos, y_instance, feature_dict)
    _, forward_score = forward(x_instance, x_instance_pos, y_vocab, feature_dict)
    return forward_score - first_term


def backward(x_instance, x_instance_pos, y_vocab, feature_dict, aggreg_fn=logsumexp):
    n, d = len(x_instance), len(y_vocab)
    scores = np.zeros( (n,d) )
    
    for i, y in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:{y}+STOP", ninf)
        e_score = feature_dict.get( f"emission:{y}+{x_instance[-1]}", ninf)
        e_score_pos = feature_dict.get( f"emission:{y}+{x_instance_pos[-1]}", ninf) 
        scores[-1, i] = t_score + e_score + e_score_pos
        
    for i in range(n-1, 0, -1):
        for y_i, y in enumerate(y_vocab):
            temp = []
            for y_next_i, y_next in enumerate(y_vocab):
                t_score = feature_dict.get( f"transition:{y}+{y_next}", ninf)
                e_score = feature_dict.get( f"emission:{y}+{x_instance[i-1]}")
                e_score_pos = feature_dict.get( f"emission:{y}+{x_instance_pos[i-1]}", ninf) 
                new_score = feature_dict.get( f"transition:{y}+{y_next}+{x_instance[i]}", ninf)
                temp.append(e_score + e_score_pos + t_score + new_score + scores[i, y_next_i])
            scores[i-1, y_i] = aggreg_fn(np.array(temp))
            
    temp = []
    for i, y_next in enumerate(y_vocab):
        t_score = feature_dict.get( f"transition:START+{y_next}")
        new_score = feature_dict.get( f"transition:START+{y_next}+{x_instance[0]}", ninf)
        temp.append(t_score + new_score + scores[0, i])
    beta = aggreg_fn(np.array(temp))
    
    return scores, beta



def forward_backward(x_instance, x_instance_pos, y_vocab, feature_dict):
    n, d = len(x_instance), len(y_vocab)
    f_scores, alpha = forward(x_instance, x_instance_pos, y_vocab, feature_dict)
    b_scores, beta = backward(x_instance, x_instance_pos, y_vocab, feature_dict)
    
    feature_expected_count = defaultdict(float)
    
    for i in range(n):
        for y_i, y in enumerate(y_vocab):
            e_feature = f"emission:{y}+{x_instance[i]}"
            e_feature_pos = f"emission:{y}+{x_instance_pos[i]}"
            e_score = feature_dict.get(e_feature, ninf) 
            e_score_pos = feature_dict.get(e_feature_pos, ninf)
            feature_expected_count[e_feature] += np.exp(f_scores[i, y_i] + b_scores[i, y_i] - e_score_pos - alpha)
            feature_expected_count[e_feature_pos] += np.exp(f_scores[i, y_i] + b_scores[i, y_i] - e_score - alpha)
    
    for i, y_next in enumerate(y_vocab):
        t_feature = f"transition:START+{y_next}"
        feature_expected_count[t_feature] += np.exp(f_scores[0, i] + b_scores[0, i] - alpha)
        t_feature = f"transition:{y_next}+STOP"
        feature_expected_count[t_feature] += np.exp(f_scores[-1, i] + b_scores[-1, i] - alpha)
        
        t_feature = f"transition:START+{y_next}+x_instance[0]"
        feature_expected_count[t_feature] += np.exp(f_scores[0, i] + b_scores[0, i] - alpha)
        
    for y_i, y in enumerate(y_vocab):
        for y_next_i, y_next in enumerate(y_vocab):
            t_feature = f"transition:{y}+{y_next}"
            t_score = feature_dict.get(t_feature, ninf)
            total, new_total = 0, 0
            for i in range(n-1):
                e_score = feature_dict.get(f"emission:{y}+{x_instance[i]}", ninf)
                e_score_pos = feature_dict.get(f"emission:{y}+{x_instance_pos[i]}", ninf)
                new_feature = f"transition:{y}+{y_next}+{x_instance[i+1]}"
                new_score = feature_dict.get(new_feature, ninf)
                prob = np.exp(f_scores[i, y_i] + b_scores[i+1, y_next_i] + t_score + e_score + e_score_pos - alpha)
                feature_expected_count[t_feature] += prob 
                feature_expected_count[new_feature] += prob

    return feature_expected_count



def get_feature_count(x_instance, x_instance_pos, y_instance, feature_dict):
    feature_count = defaultdict(int)
    
    for x, y in zip(x_instance, y_instance): feature_count[f"emission:{y}+{x}"] += 1
    for x, y in zip(x_instance_pos, y_instance): feature_count[f"emission:{y}+{x}"] += 1
    
    for y_prev, y in zip(['START'] + y_instance, y_instance + ['STOP']):
        feature_count[f"transition:{y_prev}+{y}"] += 1

    for y_prev, y, x in zip(['START']+ y_instance[:-1], y_instance, x_instance):
        feature_count[f"transition:{y_prev}+{y}+{x}"] += 1
    
    return feature_count




def gradient_fn(x_data, x_data_pos, y_data, feature_dict, y_vocab, eta=0.1):
    feature_grad = defaultdict(float)
    
    for x_instance, x_instance_pos, y_instance in zip(x_data, x_data_pos, y_data):
        feature_expected_counts = forward_backward(x_instance, x_instance_pos, y_vocab, feature_dict)
        feature_actual_counts = get_feature_count(x_instance, x_instance_pos, y_instance, feature_dict)
        for k, v in feature_expected_counts.items(): feature_grad[k] += v
        for k, v in feature_actual_counts.items(): feature_grad[k] -= v    
    
    if eta > 0: 
        for k, v in feature_dict.items(): feature_grad[k] += 2*eta*v
    
    return feature_grad
        
    
def loss_fn(x_data, x_data_pos, y_data, feature_dict, y_vocab, eta=0):
    loss = 0
    for x_instance, x_instance_pos, y_instance in zip(x_data, x_data_pos, y_data):
        loss += loss_fn_instance(x_instance, x_instance_pos, y_instance, feature_dict, y_vocab) 
    reg_loss = eta * sum([o**2 for o in feature_dict.values()]) if eta > 0 else 0
    return loss + reg_loss



# Helper function
def numpy_to_dict(weight, feature_dict):
    for i,k in enumerate(feature_dict.keys()):
        feature_dict[k] = weight[i]
    return feature_dict

def dict_to_numpy(grads, feature_dict):
    np_grads = np.zeros(len(feature_dict))
    for i, k in enumerate(feature_dict.keys()):
        np_grads[i] = grads[k]
    return np_grads

def get_loss_grad(weight, *args):
    x_data, x_data_pos, y_data, feature_dict, y_vocab = args
    feature_dict = numpy_to_dict(weight, feature_dict)
    loss = loss_fn(x_data, x_data_pos, y_data, feature_dict, y_vocab, eta=0.1)
    grads = gradient_fn(x_data, x_data_pos, y_data, feature_dict, y_vocab, eta=0.1)
    grads = dict_to_numpy(grads, feature_dict)
    return loss, grads

def callbackF(weight): print(f'Loss: \t {loss_fn(x_data, x_data_pos, y_data, feature_dict, y_vocab, eta=0.1):.4f}')



if __name__ == "__main__":
    full_dir = Path('full')
    save_dir = Path('save')
    ninf = -1e9
   
    x_data, x_data_pos, y_data = [read_data(full_dir/'train', i) for i in range(3)] 
    
    y_vocab = sorted(list(set([oo for o in y_data for oo in o])))
    x_vocab = list(set([oo for o in x_data for oo in o]))
    x_vocab_pos = sorted(list(set([oo for o in x_data_pos for oo in o])))
   
    e_dict = calc_e(x_data, y_data, x_vocab, y_vocab)
    e_dict_pos = calc_e(x_data_pos, y_data, x_vocab_pos, y_vocab)
    t_dict = calc_t(y_data, y_vocab)
    n_dict = calc_new(x_data, y_data, y_vocab)
    feature_dict = {**t_dict, **e_dict, **e_dict_pos, **n_dict}

    y_preds = inference(full_dir/'dev.in', y_vocab, feature_dict, full_dir/'dev.p2.out')
    y_label = read_data(full_dir/'dev.out', column=2)
    
    y_label = [oo for o in y_label for oo in o+['O']]
    y_preds = [oo for o in y_preds for oo in o+['O']]
   
    print('Evaluation on HMM')
    prec, rec, f1 = evaluate(y_label, y_preds, verbose=False)
    print(f'precision: {prec:.3f} \t rec: {rec:.3f} \t f1 {f1:.3f}')


    # Initialization
    init_weight = np.zeros(len(feature_dict))
    feature_dict = numpy_to_dict(init_weight, feature_dict)
    
    
    # Training
    print("Begin Training, this will take roughly 3 hours...")
    result = fmin_l_bfgs_b( 
        get_loss_grad, init_weight, pgtol=0.01, callback=callbackF,
        args=(x_data, x_data_pos, y_data, feature_dict, y_vocab) 
    )
    
    # Save weights
    feature_dict = numpy_to_dict(result[0], feature_dict)
    weight_name = save_dir/'full-part5-3.json'
    with open(weight_name, 'w') as f: json.dump(feature_dict, f)
    print(f"Weight saved to {weight_name}")


    # Inference
    with open(weight_name) as f: feature_dict = json.load(f)
    
    eval_filename = full_dir/'dev.p5.SP.out'
    y_preds = inference(full_dir/'dev.in', y_vocab, feature_dict, eval_filename)
    print(f"Wrote file to {eval_filename}")
   
    # Evaluation
    y_label = read_data(full_dir/'dev.out', column=2)
    
    y_label = [oo for o in y_label for oo in o+['O']]
    y_preds = [oo for o in y_preds for oo in o+['O']]
    
    print('Evaluation on CRF with transition and word emission, and pos emission features')
    prec, rec, f1 = evaluate(y_label, y_preds, verbose=False)
    print(f'precision: {prec:.3f} \t rec: {rec:.3f} \t f1 {f1:.3f}')
