import numpy as np
from collections import defaultdict, Counter
import conlleval
from pathlib import Path
from scipy.optimize import fmin_l_bfgs_b
import copy
ninf = -1e9

def feature_emission(y_prev, y_now, x_sent_n, W, idx):
    if idx > 0:
        if type(x_sent_n[0]) == list:
            key = f"emission:{y_prev}+{x_sent_n[0][idx-1]}"
        else:
            key = f"emission:{y_prev}+{x_sent_n[idx-1]}"
        return key
    

def feature_POS(y_prev, y_now, x_sent_n, W, idx):
    if idx > 0 and type(x_sent_n[0]) == list:
        return f"emission:{y_prev}+{x_sent_n[1][idx-1]}"


def feature_transition(y_prev, y_now, x_sent_n, W, idx):
    key = f"transition:{y_prev}+{y_now}"
    return key


def feature_combine(y_prev, y_now, x_sent_n, W, idx):
    if idx < len(x_sent_n):
        if type(x_sent_n[0]) == list:
            key = f"combine:{y_prev}+{y_now}+{x_sent_n[0][idx]}"
        else:
            key = f"combine:{y_prev}+{y_now}+{x_sent_n[idx]}"
        return key

def feature_cap(y_prev, y_now, x_sent_n, W, idx):
    if idx > 0:
        return f"emission:{y_prev}+{x_sent_n[0][idx-1][0].isupper()}"

ft, fe, fp, fc, fca = feature_transition, feature_emission, feature_POS, feature_combine, feature_cap

def compute_score(x_sent, y_sent, W, f_ls):
    score = 0
    y_sent_p = ['START'] + y_sent + ['STOP']
    n = len(y_sent) + 1
    for i in range(n):
        tag1, tag2 = y_sent_p[i], y_sent_p[i+1]
        for f in f_ls:
            f_key = f(y_prev=tag1, y_now=tag2, x_sent_n=x_sent, W=W, idx=i)
            if f_key:
                score += W.get(f_key, ninf)
    return score

def viterbi(x_instance, y_vocab, W, feature_ls):
    if type(x_instance[0]) == list:
        n, d = len(x_instance), len(y_vocab)
    else:
        n, d = len(x_instance), len(y_vocab)

    scores = np.full( (n,d), -np.inf) # initialize to be very negative
    bp = np.full( (n,d), 0, dtype=np.int)
    
    for i, y in enumerate(y_vocab):
        score = 0
        for f in feature_ls:
            f_key = f(y_prev='START', y_now=y, x_sent_n=x_instance, W=W, idx=0)
            if f_key: score += W.get(f_key, ninf)
        scores[0, i] = score
        
    for i in range(1, n):
        for y_i, y in enumerate(y_vocab):
            for y_prev_i, y_prev in enumerate(y_vocab):
                score = scores[i-1, y_prev_i]
                for f in feature_ls:
                    f_key = f(y_prev=y_prev, y_now=y, x_sent_n=x_instance, W=W, idx=i)
                    if f_key: score += W.get(f_key, ninf)
                if score > scores[i, y_i]:
                    scores[i, y_i] = score
                    bp[i, y_i] = y_prev_i
    
    final_score, final_bp = ninf, 0
    for i, y_prev in enumerate(y_vocab):
        score = scores[n-1, i]
        for f in feature_ls:
            f_key = f(y_prev=y_prev, y_now='STOP', x_sent_n=x_instance, W=W, idx=n)
            if f_key: score += W.get(f_key, ninf)

        if score > final_score: 
            final_score = score
            final_bp = i

    decoded_sequence = [ y_vocab[final_bp], ]
    for i in range(n-1, 0, -1):
        final_bp = bp[i, final_bp]
        decoded_sequence = [ y_vocab[final_bp] ] + decoded_sequence
        
    return decoded_sequence, final_score

def forward(x_instance, y_vocab, W, feature_ls):
    if type(x_instance[0]) == list:
        n, d = len(x_instance[0]), len(y_vocab)
    else:
        n, d = len(x_instance), len(y_vocab)

    scores = np.zeros( (n,d) )
    for i, y in enumerate(y_vocab):
        score = 0
        for f in feature_ls:
            f_key = f(y_prev='START', y_now=y, x_sent_n=x_instance, W=W, idx=0)
            if f_key:
                score += W.get(f_key, ninf)
        scores[0, i] = score
    
    for i in range(1, n):
        for y_i, y in enumerate(y_vocab):
            temp = copy.deepcopy(scores[i-1, :]) 
            for y_prev_i, y_prev in enumerate(y_vocab):
                score = 0
                for f in feature_ls:
                    f_key = f(y_prev=y_prev, y_now=y, x_sent_n=x_instance, W=W, idx=i)
                    if f_key:
                        score += W.get(f_key, ninf)
                temp[y_prev_i] += score
            scores[i, y_i] = logsumexp(temp)
    
    temp = copy.deepcopy(scores[-1, :])
    for i, y_prev in enumerate(y_vocab):
        score = 0
        for f in feature_ls:
            f_key = f(y_prev=y_prev, y_now='STOP', x_sent_n=x_instance, W=W, idx=n)
            if f_key:
                score += W.get(f_key, ninf)
        temp[i] += score
    alpha = logsumexp(np.array(temp))
    
    return scores, alpha


def backward(x_instance, y_vocab, W, feature_ls):
    if type(x_instance[0]) == list:
        n, d = len(x_instance[0]), len(y_vocab)
    else:
        n, d = len(x_instance), len(y_vocab)
    scores = np.zeros( (n,d) )
    
    for i, y in enumerate(y_vocab):
        s = 0
        for f in feature_ls:
            s += W.get(f(y_prev=y, y_now='STOP', x_sent_n=x_instance, W=W, idx=n), 0)
        scores[-1, i] = s
        
        
    for i in range(n-1, 0, -1):
        for y_i, y in enumerate(y_vocab):
            temp = copy.deepcopy(scores[i,:])
            for y_next_i, y_next in enumerate(y_vocab):
                s = 0
                for f in feature_ls:
                    s += W.get(f(y_prev=y, y_now=y_next, x_sent_n=x_instance, W=W, idx=i), 0)
                temp[y_next_i] += s
            scores[i-1, y_i] = logsumexp(np.array(temp))
            
    temp = copy.deepcopy(scores[0,:])
    for i, y_next in enumerate(y_vocab):
        s = 0
        for f in feature_ls:
            s += W.get(f(y_prev='START', y_now=y_next, x_sent_n=x_instance, W=W, idx=0), 0)
        temp[i] += s
    beta = logsumexp(np.array(temp))
    
    return scores, beta


def forward_backward(x_instance, y_vocab, W, tf_ls, sf_ls):
    if type(x_instance[0]) == list:
        n, d = len(x_instance[0]), len(y_vocab)
    else:
        n, d = len(x_instance), len(y_vocab)
    f_scores, alpha = forward(x_instance, y_vocab, W, feature_ls=[*tf_ls, *sf_ls])
    b_scores, beta = backward(x_instance, y_vocab, W, feature_ls=[*tf_ls, *sf_ls])
    
    feature_expected_count = defaultdict(float)
    
    for i in range(n):
        for y_i, y in enumerate(y_vocab):
            
            for sf in sf_ls:
                s_key = sf(y_prev=y, y_now=None, x_sent_n=x_instance, W=W, idx=i+1)
                if s_key:
                    feature_expected_count[s_key] += np.exp(f_scores[i, y_i] + b_scores[i, y_i] - alpha)
    
    for i in range(n+1):
        if i == 0:
            for yi, y in enumerate(y_vocab):
                for f in tf_ls:
                    key = f(y_prev='START', y_now=y, x_sent_n=x_instance, W=W, idx=0)
                    if key:
                        feature_expected_count[key] += np.exp(f_scores[i, yi] + b_scores[i, yi] - alpha)
        elif i == n:
            for yi, y in enumerate(y_vocab):
                for f in tf_ls:
                    key = f(y_prev=y, y_now='STOP', x_sent_n=x_instance, W=W, idx=n)
                    if key:
                        feature_expected_count[key] += np.exp(f_scores[n-1, yi] + b_scores[n-1, yi] - alpha)
        else:
            for y_prev_i, y_prev in enumerate(y_vocab):
                for y_now_i, y_now in enumerate(y_vocab):
                    
                    tkey_ls = []
                    t_score = 0
                    for tf in tf_ls:
                        t_key = tf(y_prev=y_prev, y_now=y_now, x_sent_n=x_instance, W=W, idx=i)
                        if t_key:
                            t_score += W.get(t_key, ninf)
                            tkey_ls.append(t_key)
                    s_score = 0
                    for sf in sf_ls:
                        s_key = sf(y_prev=y_prev, y_now=y_now, x_sent_n=x_instance, W=W, idx=i)
                        if s_key:
                            s_score += W.get(s_key, ninf)
                        
                    for t_key in tkey_ls:
                        feature_expected_count[t_key] += \
                            np.exp(f_scores[i-1, y_prev_i] + b_scores[i, y_now_i] + t_score + s_score - alpha)
            
    return feature_expected_count

def get_feature_count(x_sent, y_sent, f_ls):

    feature_count = defaultdict(int)
    y_sent_p = ['START'] + y_sent + ['STOP']
    
    n = len(y_sent) + 1
    for i in range(n):
        tag1, tag2 = y_sent_p[i], y_sent_p[i+1]
        for f in f_ls:
            f_key = f(y_prev=tag1, y_now=tag2, x_sent_n=x_sent, W=None, idx=i)
            if f_key:
                feature_count[f_key] += 1
    return feature_count

def logsumexp(a):
    b = a.max()
    return  b + np.log( (np.exp(a-b)).sum() )


def loss_fn_instance(x_instance, y_instance, feature_dict, y_vocab, f_ls):
    first_term = compute_score(x_instance, y_instance, feature_dict, f_ls)
    _, forward_score = forward(x_instance, y_vocab, feature_dict, f_ls)
    return forward_score - first_term
        
    
def loss_fn(x_data, y_data, feature_dict, y_vocab, f_ls, eta=0):
    loss = 0
    for x_instance, y_instance in zip(x_data, y_data):
        loss += loss_fn_instance(x_instance, y_instance, feature_dict, y_vocab, f_ls) 
    
    reg_loss = eta * sum([o**2 for o in feature_dict.values()])
    return loss + reg_loss


def gradient_fn(x_data, y_data, W, y_vocab, tf_ls, sf_ls, eta=0.1):
    feature_grad = defaultdict(float)
    
    for x_instance, y_instance in zip(x_data, y_data):
        feature_expected_counts = forward_backward(x_instance, y_vocab, W, tf_ls, sf_ls)
        feature_actual_counts = get_feature_count(x_instance, y_instance, [*tf_ls, *sf_ls])
        for k, v in feature_expected_counts.items(): 
            feature_grad[k] += v
        for k, v in feature_actual_counts.items(): 
            feature_grad[k] -= v    
    
    if eta > 0: 
        for k, v in W.items(): feature_grad[k] += 2*eta*v
    return feature_grad



def get_loss_grad(weight, *args):
    X, Y_NER, W, Y_NER_set, tf_ls, sf_ls = args
    feature_dict = numpy_to_dict(weight, W)
    loss = loss_fn(X, Y_NER, W, Y_NER_set, [*tf_ls, *sf_ls], eta=0.1)
    print(loss)
    grads = gradient_fn(X, Y_NER, W, Y_NER_set, tf_ls, sf_ls, eta=0.1)
    grads = dict_to_numpy(grads, W)
    return loss, grads


def read_file(path, column=0):
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



def get_set(mlist):
    '''
    Assmue mlist is a 2-d list with STRING elements
    '''
    output = set()
    for line in mlist:
        output.update(line)
    return sorted(output)


def write_result(file_ptr, line): file_ptr.write(line+'\n')


def numpy_to_dict(weight, feature_dict):
    for i,k in enumerate(feature_dict.keys()):
        feature_dict[k] = weight[i]
    return feature_dict

def dict_to_numpy(grads, feature_dict):
    np_grads = np.zeros(len(feature_dict))
    for i, k in enumerate(feature_dict.keys()):
        np_grads[i] = grads[k]
    return np_grads


def new_X(X, POS):
    new_x = []

    for x_sent, pos_sent in zip(X, POS):
        new_x.append([x_sent, pos_sent])
    return new_x


def callbackF(weight): print(f'Loss: \t {loss_fn(X_new, Y_NER_train, W, Y_NER_set, f_ls, eta=0.1):.4f}')

if __name__ == "__main__":
    X_train = read_file("partial/train", 0)
    Y_NER_train = read_file("partial/train", 1)
    Y_POS_train = read_file("full/train", 1)
    X_new = new_X(X_train, Y_POS_train)
    X_set = get_set(X_train)
    Y_NER_set = get_set(Y_NER_train)
    Y_POS_set = get_set(Y_POS_train)
    X_cap_set = [True, False]

    f_ls = [fe, ft, fp, fc, fca]
    sf_ls = [fe, fp, fca]
    tf_ls = [ft, fc]


    # Initialization

    W = dict()
    for y_prev in Y_NER_set:
        
        ft_start_key = ft(y_prev='START', y_now=y_prev, x_sent_n=None, W=None, idx=0)
        W[ft_start_key] = 0
        ft_stop_key = ft(y_prev=y_prev, y_now='STOP', x_sent_n=None, W=None, idx=0)
        W[ft_stop_key] = 0

        for y_now in Y_NER_set:
            f_key = ft(y_prev=y_prev, y_now=y_now, x_sent_n=None, W=None, idx=0)
            W[f_key] = 0
            for x in X_set:
                fc_key = fc(y_prev=y_prev, y_now=y_now, x_sent_n=[x], W=None, idx=0)
                W[fc_key] = 0

                fe_key = fe(y_prev=y_prev, y_now=None, x_sent_n=[x], W=None, idx=1)
                W[fe_key] = 0

                fca_key = fca(y_prev=y_prev, y_now=None, x_sent_n=[x], W=None, idx=1)
                W[fca_key] = 0

        for pos in Y_POS_set:
            f_key = fe(y_prev=y_prev, y_now=None, x_sent_n=[pos], W=None, idx=1)
            W[f_key] = 0

    init_weight = np.zeros(len(W))
    feature_dict = numpy_to_dict(init_weight, W)

    # Training
    print("Begin Training, this will take roughly 3 hours...")

    optimal_w, final_loss, _ = fmin_l_bfgs_b( 
        get_loss_grad, init_weight, pgtol=0.05, callback=callbackF,
        args=(X_new, Y_NER_train, W, Y_NER_set, tf_ls, sf_ls) 
    )
    


    # Save weights
    feature_dict = numpy_to_dict(optimal_w, W)
    weight_name = 'save/p6-1.json'
    with open(weight_name, 'w') as f: json.dump(W, f)
    print(f"Weight saved to {weight_name}")

    # load weights
    weight_name = 'save/p5-2.json'
    with open(weight_name) as f: 
        feature_dict = json.load(f)
    W = feature_dict
    
    X_dev = read_file("partial/dev.in", 0)
    Y_dev = read_file("partial/dev.out", 1)

    save_path = "test.p6.CRF.f4.out"
    f = open(save_path, 'w')
    evals = []

    # predict
    for i in range(len(X_dev)):
        Y_pred, _ = viterbi(X_dev[i], Y_NER_set, W, f_ls)
        for j in range(len(Y_pred)):
            line = "{} {} {}".format(X_dev[i][j], Y_dev[i][j], Y_pred[j])
            f.write(line+'\n')
            evals.append(line)
        f.write('\n')
    res = conlleval.evaluate(evals)
    print(conlleval.report(res))
    f.close()


