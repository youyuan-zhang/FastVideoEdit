import copy

import torch


def find_sub(full, sub, full_len, sub_len, min_idx):
    if sub_len == 0:
        return -1
    for full_idx in range(min_idx, full_len - sub_len + 2):
        if full[full_idx] == sub[1]:
            flag = True
            for sub_idx in range(sub_len):
                if full[full_idx + sub_idx] != sub[1 + sub_idx]:
                    flag = False
                    break
            if flag:
                return full_idx
    return -1


def generate_alpha(full, sub_idx, sub_len, max_len):
    alpha = torch.zeros(max_len, dtype=torch.int64)
    if sub_len == 0 or sub_idx == -1:
        return alpha
    alpha[sub_idx: sub_idx + sub_len] = 1
    return alpha


def clean_seq(full, sub_idx, full_len, sub_len):
    if sub_len == 0 or sub_idx == -1:
        return full[:], full_len
    return full[:sub_idx] + full[sub_idx + sub_len:], full_len - sub_len


def get_refinement_mapper(prompts, specifiers, change, smask, tokenizer, encoder, device, max_len=77):
    source_prompt = prompts[0]
    target_prompt = prompts[1]
    change_list = change
    smask_list = smask
    local_list = specifiers[0][0]
    mutual_list = specifiers[0][1]

    x_seq = tokenizer.encode(source_prompt)
    y_seq = tokenizer.encode(target_prompt)
    c_seq_list = [tokenizer.encode(change) for change in change_list]
    s_seq_list = [tokenizer.encode(smask) for smask in smask_list]
    e_seq_list = [tokenizer.encode(local) for local in local_list]
    m_seq_list = [tokenizer.encode(mutual) for mutual in mutual_list]

    x_len = len(x_seq) - 2
    y_len = len(y_seq) - 2
    c_len_list = [len(c_seq) - 2 for c_seq in c_seq_list]
    s_len_list = [len(s_seq) - 2 for s_seq in s_seq_list]
    e_len_list = [len(e_seq) - 2 for e_seq in e_seq_list]
    m_len_list = [len(m_seq) - 2 for m_seq in m_seq_list]

    min_idx = 1
    c_idx_list = []
    for c_seq, c_len in zip(c_seq_list, c_len_list):
        c_idx_list.append(find_sub(x_seq, c_seq, x_len, c_len, min_idx))
        min_idx = max(max(c_idx_list), min_idx)
    min_idx = 1
    s_idx_list = []
    for s_seq, s_len in zip(s_seq_list, s_len_list):
        s_idx_list.append(find_sub(x_seq, s_seq, x_len, s_len, min_idx))
        min_idx = max(max(s_idx_list), min_idx)
    min_idx = 1
    e_idx_list = []
    for e_seq, e_len in zip(e_seq_list, e_len_list):
        e_idx_list.append(find_sub(y_seq, e_seq, y_len, e_len, min_idx))
        min_idx = max(max(e_idx_list), min_idx)
    min_idx = 1
    m_idx_list = []
    for m_seq, m_len in zip(m_seq_list, m_len_list):
        m_idx_list.append(find_sub(y_seq, m_seq, y_len, m_len, min_idx))
        min_idx = max(max(m_idx_list), min_idx)

    alpha_c_list = [generate_alpha(source_prompt, c_idx, c_len, max_len) for c_idx, c_len in zip(c_idx_list, c_len_list)]
    alpha_s_list = [generate_alpha(source_prompt, s_idx, s_len, max_len) for s_idx, s_len in zip(s_idx_list, s_len_list)]
    alpha_e_list = [generate_alpha(target_prompt, e_idx, e_len, max_len) for e_idx, e_len in zip(e_idx_list, e_len_list)]
    alpha_m_list = [generate_alpha(target_prompt, m_idx, m_len, max_len) for m_idx, m_len in zip(m_idx_list, m_len_list)]

    alpha_c = (torch.stack(alpha_c_list).sum(0) > 0).to(torch.int64)
    alpha_s = (torch.stack(alpha_s_list).sum(0) > 0).to(torch.int64)
    alpha_e = (torch.stack(alpha_e_list).sum(0) > 0).to(torch.int64)
    alpha_m = (torch.stack(alpha_m_list).sum(0) > 0).to(torch.int64)
    alphas = 1 - alpha_e
    ms = copy.deepcopy(alphas)

    x_clean, x_clean_len = x_seq, x_len
    for i, c in enumerate(zip(c_idx_list, c_len_list)):
        c_idx, c_len = c
        x_clean, x_clean_len = clean_seq(x_clean, c_idx - sum(c_len_list[:i]), x_clean_len, c_len)
    y_clean, y_clean_len = y_seq, y_len
    for i, e in enumerate(zip(e_idx_list, e_len_list)):
        e_idx, e_len = e
        y_clean, y_clean_len = clean_seq(y_clean, e_idx - sum(e_len_list[:i]), y_clean_len, e_len)
    
    if not (x_clean == y_clean and x_clean_len == y_clean_len):
        print(f"source_prompt: {x_seq}")
        print(f"target_prompt: {y_seq}")
        print(f"change: {c_seq_list}")
        print(f"local: {e_seq_list}")
        print(f"mutual: {m_seq_list}")
        print(f"change: {c_idx_list}")
        print(f"local: {e_idx_list}")
        print(f"mutual: {m_idx_list}")
        print(f"alpha_c: {alpha_c}")
        print(f"alpha_e: {alpha_e}")
        print(f"alpha_m: {alpha_m}")
        print(f"alphas: {alphas}")
        print(f"ms: {ms}")
        assert False

    mapper = torch.arange(0, max_len, dtype=torch.int64)
    j = 0
    for i in range(max_len):
        if alpha_c[i] == 1:
            mapper[j:] += 1
            j -= 1
        j += 1
    for i in range(max_len):
        if alpha_e[i] == 1:
            mapper[i] = -1
            mapper[i + 1:] -= 1
    mapper[mapper >= max_len] = max_len - 1
    
    if alpha_s.sum() > 0:
        alpha_c = alpha_s

    # print(f"source_prompt: {source_prompt}")
    # print(f"target_prompt: {target_prompt}")
    # print(f"change: {change_list}")
    # print(f"local: {local_list}")
    # print(f"mutual: {mutual_list}")
    #
    # print(f"source_prompt: {x_seq}")
    # print(f"target_prompt: {y_seq}")
    # print(f"change: {c_seq_list}")
    # print(f"local: {e_seq_list}")
    # print(f"mutual: {m_seq_list}")
    #
    # print(f"source_prompt: {x_len}")
    # print(f"target_prompt: {y_len}")
    # print(f"change: {c_len_list}")
    # print(f"local: {e_len_list}")
    # print(f"mutual: {m_len_list}")
    #
    # print(f"change: {c_idx_list}")
    # print(f"local: {e_idx_list}")
    # print(f"mutual: {m_idx_list}")
    #
    # print(f"alpha_c: {alpha_c}")
    # print(f"alpha_e: {alpha_e}")
    # print(f"alpha_m: {alpha_m}")
    # print(f"alphas: {alphas}")
    # print(f"ms: {ms}")
    #
    # print(f"x_clean: {x_clean}")
    # print(f"y_clean: {y_clean}")
    # print(f"x_clean_len: {x_clean_len}")
    # print(f"y_clean_len: {y_clean_len}")
    
    # print(f"mapper: {mapper}")

    return (mapper.unsqueeze(0),
            alphas.unsqueeze(0), ms.unsqueeze(0),
            alpha_c.unsqueeze(0), alpha_e.unsqueeze(0), alpha_m.unsqueeze(0))
