def get_seed(prompt, i, seed):
    h = 0
    for el in prompt:
        h += ord(el)
    h += i
    return h + seed