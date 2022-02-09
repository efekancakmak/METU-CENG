def utility_tictactoe(state, depth):
    ## check win states...
    if  state[0][0] + state[1][1] + state[2][2] == 'XXX' or \
        state[0][0] + state[0][1] + state[0][2] == 'XXX' or \
        state[1][0] + state[1][1] + state[1][2] == 'XXX' or \
        state[2][0] + state[2][1] + state[2][2] == 'XXX' or \
        state[0][0] + state[1][0] + state[2][0] == 'XXX' or \
        state[0][1] + state[1][1] + state[2][1] == 'XXX' or \
        state[0][2] + state[1][2] + state[2][2] == 'XXX' or \
        state[2][0] + state[1][1] + state[0][2] == 'XXX':
        return 5-0.01*(depth-1)
    ## check lose states
    if  state[0][0] + state[1][1] + state[2][2] == 'OOO' or \
        state[0][0] + state[0][1] + state[0][2] == 'OOO' or \
        state[1][0] + state[1][1] + state[1][2] == 'OOO' or \
        state[2][0] + state[2][1] + state[2][2] == 'OOO' or \
        state[0][0] + state[1][0] + state[2][0] == 'OOO' or \
        state[0][1] + state[1][1] + state[2][1] == 'OOO' or \
        state[0][2] + state[1][2] + state[2][2] == 'OOO' or \
        state[2][0] + state[1][1] + state[0][2] == 'OOO':
        return -5
    ## check whether it is tied
    tied = True
    for i in range(3):
        for j in range(3):
            if state[i][j] == ' ':
                tied = False
    if tied==True:
        return 0
    ## or keep playing..
    ## I select number 6 that is out of utility
    ## it is simply signal indicating "I dont know my utility.."
    return 6

def value_tictactoe(state, depth, ismax, path, AlpBet, alpha, beta):
    util = utility_tictactoe(state,depth)
    if util != 6:
        return util
    if ismax == True:
        return max_tictactoe(state,depth,path,AlpBet,alpha,beta)
    else:
        return min_tictactoe(state,depth,path,AlpBet,alpha,beta)

def max_tictactoe(state, depth, path, AlpBet, alpha, beta):
    """
    firstcall of this issue creates
    a move for root state..
    So that I check whether is first call
    by assigning path = ["first"] and checking it.
    """
    firstcall = False
    k, l = 0,0
    if len(path)!=0 and path[0] == "first": # firstcall.. need to save move
        firstcall = True
        path.pop()
    v = -99999999999999999999
    for i in range(3):
        for j in range(3):
            if state[i][j] == ' ':
                # Deep copy of current state
                successor = [[state[0][0],state[0][1],state[0][2]], [state[1][0],state[1][1],state[1][2]], [state[2][0],state[2][1],state[2][2]]]
                # Filling empty slot with 'X'
                successor[i][j] = 'X'
                # Add to the path
                path.append(successor)
                vtemp = value_tictactoe(successor,depth+1,False,path,AlpBet,alpha,beta)
                if vtemp > v:
                    k,l = i,j
                    v = vtemp
                if AlpBet == True:
                    if v >= beta:
                        #this 'if' below aims to secure edge cases unexpected     
                        if firstcall==True:
                            path.append((l,k))
                        return v
                    
                    alpha = max(alpha,v)
    # if it is first call, I save the move.
    if firstcall==True:
        path.append((l,k))
    return v

def min_tictactoe(state, depth, path, AlpBet, alpha, beta):
    v = 99999999999999999999
    for i in range(3):
        for j in range(3):
            if state[i][j] == ' ':
                successor = [[state[0][0],state[0][1],state[0][2]], [state[1][0],state[1][1],state[1][2]], [state[2][0],state[2][1],state[2][2]]]
                successor[i][j] = 'O'
                path.append(successor)
                v = min(v, value_tictactoe(successor,depth+1,True,path,AlpBet,alpha,beta))
                if AlpBet == True:
                    if v <= alpha:
                        return v
                    beta = min(beta,v)
    return v

def tictactoe(method_name, problem_file_name):
    # Read input
    f = open(problem_file_name,'r')
    s = f.read().split('\n')
    f.close()
    for i in range(3):
        s[i] = list(s[i])[:-1]
    # Generate path with 'first' sign
    # It'll be used to decide move in the first call
    path = ['first']
    alpha = -9999999999999
    beta = 99999999999999
    if method_name == "Minimax":
        res = value_tictactoe(s,0,True,path,False,alpha,beta)
    else:
        res = value_tictactoe(s,0,True,path,True,alpha,beta)
    move = path[-1]
    # if we are already at terminal state
    if path==["first"]:
        return res,[],[]
    path = [i[0][0]+i[0][1]+i[0][2]+i[1][0]+i[1][1]+i[1][2]+i[2][0]+i[2][1]+i[2][2] for i in path[:-1]]
    return res,move,path

############################## Game Tree Begins ################################################

def max_gametree(state, utilities, path, successors, AlpBet, alpha, beta):
    """
    they are almost the same with the tic-tac-toe functions
    so that I mention for just different lines.
    """
    firstcall = False
    if len(path)!=0 and path[0] == "first": # firstcall.. need to save move
        firstcall = True
        path.pop()
    v = -9999999999
    move = ''
    for s in successors[state]:
        # in successors dict
        # a record is like <state>:[<successor>,...]
        # so that s is like (<succesor's state>,<edge>)
        path.append(s[0])
        vtemp =value_gametree(s[0],utilities,path,False,successors,AlpBet,alpha,beta)
        if vtemp > v:
            v = vtemp
            move = s[1]
        if AlpBet==True:
            if v>=beta:
                if firstcall==True:
                    path.append(move)
                return v
            alpha = max(alpha,v)
    if firstcall==True:
        path.append(move)
    return v


def min_gametree(state, utilities, path, successors, AlpBet, alpha, beta):
    firstcall = False
    if len(path)!=0 and path[0] == "first": # firstcall.. need to save move
        firstcall = True
        path.pop()
    v = 9999999999
    move = ''
    for s in successors[state]:
        path.append(s[0])
        vtemp = value_gametree(s[0],utilities,path,True,successors,AlpBet,alpha,beta)
        if vtemp < v:
            v = vtemp
            move = s[1]
        if AlpBet==True:
            if v<=alpha:
                if firstcall==True:
                    path.append(move)
                return v
            beta = min(beta,v)
    if firstcall==True:
        path.append(move)
    return v
        
def value_gametree(state, utilities, path, ismax, successors, AlpBet, alpha, beta):
    if state in utilities:
        return int(utilities[state])
    if ismax == True:
        return max_gametree(state,utilities,path,successors,AlpBet,alpha,beta)
    else:
        return min_gametree(state,utilities,path,successors,AlpBet,alpha,beta)
    
def gameTree(method_name,problem_file_name,player_type):
    # Reading input
    f = open(problem_file_name,'r')
    l = f.readlines()
    f.close()
    l = [i[:-1] for i in l[:-1]] + [l[-1]]
    root = l[0]
    # successors are saved in a dict
    successors = {}
    # leaf values is saved in a dict
    utilities = {}
    path = ["first"]
    for i in l[1:]:
        if ' ' in i:
            tmp = i.split(' ')
            if tmp[0] not in successors:
                successors[tmp[0]] = [(tmp[1],tmp[2])]
            else:
                successors[tmp[0]].append((tmp[1],tmp[2]))
        else:
            tmp = i.split(':')
            utilities[tmp[0]] = tmp[1]
    if player_type == "MAX":
        if method_name == "Minimax":
            res = value_gametree(root,utilities,path,True,successors,False,0,0)
        else:
            res = value_gametree(root,utilities,path,True,successors,True,-9999999999,9999999999)
    elif player_type == "MIN":
        if method_name == "Minimax":
            res = value_gametree(root,utilities,path,False,successors,False,0,0)
        else:
            res = value_gametree(root,utilities,path,False,successors,True,-9999999999,9999999999)
    # if we are already at terminal state
    if path == ["first"]:
        return res,[],[]
    return res,path[-1],path[:-1]


def SolveGame(method_name, problem_file_name, player_type):
    if problem_file_name[:8]=="gametree":
        return gameTree(method_name,problem_file_name,player_type)
    elif problem_file_name[:9]=="tictactoe":
        return tictactoe(method_name,problem_file_name)
    else:
        print("Wrong text file...")
    return 0