def viterbi(problem_file_name):
    # reading input...
    f = open(problem_file_name,'r')
    l = f.read()
    l = l.split('\n')
    f.close()
    states = l[1].split('|')
    temp = l[3].split('|')
    sp = {}
    for p in temp:
        x = p.split(':')
        sp[x[0]] = eval(x[1])
    temp = l[5].split('|')
    tp = {}
    for p in temp:
        x = p.split(':')
        y = x[0].split('-')
        tp[(y[0],y[1])] = eval(x[1])
    temp = l[7].split('|')
    op = {}
    for p in temp:
        x = p.split(':')
        y = x[0].split('-')
        op[(y[0],y[1])] = eval(x[1])
    observations = l[9].split('|')
    ## ## ## ## ## ## ## ## ## ## ## ##
    N = len(states)
    T = len(observations)
    # init matrices..
    viterbi, back = {}, {}
    for s in states:
        viterbi[s] = [0]*T
        back[s] = [0]*T
    # initial states should be set manually
    for s in states:
        viterbi[s][0] = sp[s] * op[s,observations[0]]
        back[s][0] 
    # after initial, viterbi filling the matrix
    # by using dynamic-programming
    for t in range(1,T):
        for s in states:
            mAx = -99**9
            argmax = -1
            for si in states:
                temp = viterbi[si][t-1] * tp[si,s] * op[s,observations[t]]
                if temp > mAx:
                    mAx = temp
                    argmax = si
            viterbi[s][t] = mAx
            back[s][t] = argmax
    # extract information from matrix
    # then select most possible sequence result.
    best = [[row,viterbi[row][T-1]] for row in viterbi]
    mAx = -99**9
    arg = -1
    for i in best:
        if i[1] > mAx:
            mAx = i[1]
            arg = i[0]
    # by doing backtracking, find the path.
    path = [arg]
    t = T-1
    while t > 0:
        path.append(back[arg][t])
        best = path[-1]
        t -= 1 

    return path[::-1], mAx, viterbi
