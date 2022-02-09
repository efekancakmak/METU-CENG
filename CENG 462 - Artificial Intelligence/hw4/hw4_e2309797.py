from copy import deepcopy

def equals(a,b,epsilon):
    n = len(a)
    m = len(a[0])
    for i in range(n):
        for j in range(m):
            if abs(a[i][j] - b[i][j]) > epsilon:
                return False
    return True

def SolveMDP (method_name , problem_file_name ):
    # Reading input file..
    f = open(problem_file_name,'r')
    l = f.readlines()
    f.close()
    environment = (eval(l[1].split(' ')[0]),eval(l[1].split(' ')[1]))
    obstacles = [eval(i) for i in l[3].split('|')]
    goals = eval('{'+ l[5].replace('|',',') + '}')
    reward = eval(l[7])
    action_noise = [eval(l[9]), eval(l[10]), eval(l[11])]
    gamma = eval(l[13])
    epsilon = eval(l[15])
    iteration = eval(l[17])
    if method_name == "ValueIteration":
        # all expected values equal to 0 at first..
        # n x m matrices to save utility for states..
        util_k0 = [[1 for i in range(environment[1])] for j in range(environment[0])]
        util_k1 = [[0 for i in range(environment[1])] for j in range(environment[0])]
        # a dict to save optimal policy..
        policy = {}
        while not equals(util_k0,util_k1,epsilon):
            util_k0 = deepcopy(util_k1)
            for i in range(environment[0]):
                for j in range(environment[1]):
                    # for each state..
                    # determine current state situation..
                    if (i,j) in obstacles:
                        util_k1[i][j] = 0
                        util_k0[i][j] = 0
                        continue
                    elif (i,j) in goals:
                        util_k1[i][j] = goals[(i,j)]
                        util_k0[i][j] = goals[(i,j)]
                        continue
                    else:
                        here = reward
                    
                    # check whether right move can be done
                    # and calculate q-state utility of actions
                    if j == environment[1]-1 or (i,j+1) in obstacles:
                        right =  here + gamma*util_k0[i][j]
                    elif (i,j+1) in goals:
                        right = here + gamma*goals[(i,j+1)]
                    else:
                        right = reward + gamma*util_k0[i][j+1]

                    # check whether down move can be done
                    # and calculate q-state utility of actions
                    if i == environment[0]-1 or (i+1,j) in obstacles:
                        down =  here + gamma*util_k0[i][j]
                    elif (i+1,j) in goals:
                        down = here + gamma*goals[(i+1,j)]
                    else:
                        down = reward + gamma*util_k0[i+1][j]

                    # check whether left move can be done
                    # and calculate q-state utility of actions
                    if j == 0 or (i,j-1) in obstacles:
                        left = here + gamma*util_k0[i][j]
                    elif (1,j-1) in goals:
                        left = here + gamma*goals[(i,j-1)]
                    else:
                        left = reward + gamma*util_k0[i][j-1]
                    
                    # check whether up move can be done
                    # and calculate q-state utility of actions
                    if i == 0 or (i-1,j) in obstacles:
                        up = here + gamma*util_k0[i][j]
                    elif (i-1,j) in goals:
                        up = here + gamma*goals[(i-1,j)]
                    else:
                        up = reward + gamma*util_k0[i-1][j]
                    
                    # calculate weighted sum of actions with action noises
                    act = []
                    act.append((action_noise[1]*(up) + action_noise[0]*(right) + action_noise[2]*(down) , '>'))
                    act.append((action_noise[1]*(right) + action_noise[0]*(down) + action_noise[2]*(left) , 'V'))
                    act.append((action_noise[1]*(down) + action_noise[0]*(left) + action_noise[2]*(up) , '<'))
                    act.append((action_noise[1]*(left) +  action_noise[0]*(up) + action_noise[2]*(right) , '^'))
                    # sort them and take the action that has maximum utility
                    act.sort(key=lambda x: x[0],reverse=True)
                    util_k1[i][j] = act[0][0]
                    policy[(i,j)] = act[0][1]
        # IO issues..
        util = {}
        for i in range(environment[0]):
            for j in range(environment[1]):
                util[(i,j)] = round(util_k1[i][j],2)
        return util, policy
    elif method_name == "PolicyIteration":
        util_k0 = [[0 for i in range(environment[1])] for j in range(environment[0])]
        util_k1 = [[0 for i in range(environment[1])] for j in range(environment[0])]
        policy = {}
        # generate initial policy
        # by assigning all results of policy to right '>'
        for i in range(environment[0]):
            for j in range(environment[1]):
                if (i,j) not in goals and (i,j) not in obstacles:
                    policy[(i,j)] = '>'
        unchanged = False
        while not unchanged:
            # policy evaluation
            for k in range(iteration):
                util_k0 = deepcopy(util_k1)
                for i in range(environment[0]):
                    for j in range(environment[1]):
                        # detect situation of the state
                        if (i,j) in obstacles:
                            util_k1[i][j] = 0
                            continue
                        elif (i,j) in goals:
                            util_k1[i][j] = goals[(i,j)]
                            continue
                        else:
                            r = reward
                        # these if else lines are almost the same with Value Iteration part.
                        if j == environment[1]-1 or (i,j+1) in obstacles:
                            right = r + gamma*util_k0[i][j]
                        elif (i,j+1) in goals:
                            right = r + gamma*goals[(i,j+1)]
                        else:
                            right = r + gamma*util_k0[i][j+1]

                        if i == environment[0]-1 or (i+1,j) in obstacles:
                            down = r + gamma*util_k0[i][j]
                        elif (i+1,j) in goals:
                            down = r + gamma*goals[(i+1,j)]
                        else:
                            down = r + gamma*util_k0[i+1][j]

                        if j == 0 or (i,j-1) in obstacles:
                            left = r + gamma*util_k0[i][j]
                        elif (1,j-1) in goals:
                            left = r + gamma*goals[(i,j-1)]
                        else:
                            left = r + gamma*util_k0[i][j-1]
                        
                        if i == 0 or (i-1,j) in obstacles:
                            up = r + gamma*util_k0[i][j]
                        elif (i-1,j) in goals:
                            up = r + gamma*goals[(i-1,j)]
                        else:
                            up = r + gamma*util_k0[i-1][j]

                        # we do what previous policy says..
                        if policy[(i,j)] == '>':
                            res = (action_noise[1]*(up) + action_noise[0]*(right) + action_noise[2]*(down))
                        elif policy[(i,j)] == 'V':
                            res = (action_noise[1]*(right) + action_noise[0]*(down) + action_noise[2]*(left))
                        elif policy[(i,j)] == '<':
                            res = (action_noise[1]*(down) + action_noise[0]*(left) + action_noise[2]*(up))
                        else:
                            res = (action_noise[1]*(left) +  action_noise[0]*(up) + action_noise[2]*(right))
                        
                        util_k1[i][j] = res 

            # then, use evaluated policy and try to fix it...
            unchanged = True
            for i in range(environment[0]):
                for j in range(environment[1]):
                    # for each state...
                    # detect the state: 
                    #obstacle or goals dont have a action,  so didnot play with them..      
                    if (i,j) in obstacles:
                        continue
                    elif (i,j) in goals:
                        continue
                    else:
                        here = reward
                    
                    ## again almost same lines with Value Iteration
                    if j == environment[1]-1 or (i,j+1) in obstacles:
                        right =  here + gamma*util_k0[i][j]
                    elif (i,j+1) in goals:
                        right = here + gamma*goals[(i,j+1)]
                    else:
                        right = here + gamma*util_k0[i][j+1]

                    if i == environment[0]-1 or (i+1,j) in obstacles:
                        down =  here + gamma*util_k0[i][j]
                    elif (i+1,j) in goals:
                        down = here + gamma*goals[(i+1,j)]
                    else:
                        down = here + gamma*util_k0[i+1][j]

                    if j == 0 or (i,j-1) in obstacles:
                        left = here + gamma*util_k0[i][j]
                    elif (1,j-1) in goals:
                        left = here + gamma*goals[(i,j-1)]
                    else:
                        left = here + gamma*util_k0[i][j-1]
                    
                    if i == 0 or (i-1,j) in obstacles:
                        up = here + gamma*util_k0[i][j]
                    elif (i-1,j) in goals:
                        up = here + gamma*goals[(i-1,j)]
                    else:
                        up = here + gamma*util_k0[i-1][j]
                    
                    # calculate weighted sum of actions with their action noises
                    act = []
                    act.append((action_noise[1]*(up) + action_noise[0]*(right) + action_noise[2]*(down) , '>'))
                    act.append((action_noise[1]*(right) + action_noise[0]*(down) + action_noise[2]*(left) , 'V'))
                    act.append((action_noise[1]*(down) + action_noise[0]*(left) + action_noise[2]*(up) , '<'))
                    act.append((action_noise[1]*(left) +  action_noise[0]*(up) + action_noise[2]*(right) , '^'))
                    # sort them to take one having maximum utility
                    act.sort(key=lambda x: x[0],reverse=True)

                    # if any action is better than in previous policy
                    #        -> fix this policy 
                    if act[0][0] > util_k1[i][j]:
                        # fix
                        policy[(i,j)] = act[0][1]
                        # continue to fixing
                        unchanged = False
        # IO issues...
        util = {}
        for i in range(environment[0]):
            for j in range(environment[1]):
                util[(i,j)] = round(util_k1[i][j],2)
        return util, policy
    else:
        print("Wrong method name..")
    return 0