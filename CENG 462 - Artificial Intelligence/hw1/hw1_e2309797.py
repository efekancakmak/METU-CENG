def post_process(source, destination, solution, cost):
    """
    AS I MENTIONED IN bFs, dLs, uCs and
    iDf, I STORE NODES IN DIFFERENT STRUCTURES..
    BECAUSE OF THAT, THIS POST FUNCTION EXRACTS DESIRED OUTPUT FROM MY RAW DATA...
    """
    path = []
    i = 0
    j = 0
    ### solution list now is like below since I used solution list as queue too.
    ### [ processed1, processed2, ..., processedx (destination), explored1, explored2, ...]
    while j<len(solution):
        if solution[j][0]==destination:
            path.append(destination)
            i = solution[j][1]
            break
        j+=1
    solution = solution[:j+1]
    ### solution list now is like below
    ### [ processed1, processed2, ..., processedx (destination)]

    ### Doing backtracking using saved parent index data
    depth = 0
    while True:
        path = [solution[i][0]] + path
        i = solution[i][1]
        depth += 1
        if i == -1:
            break

    ### Deleting unnecesarry parts...
    ### solution list now is like below
    ### [ [<processed1>, <parent>, <cost>], [<processed2>, <parent>, <cost>], ... ]
    i = 0
    while i<=j:
        solution[i] = solution[i][0]
        i+=1
    ### solution list now is like below
    ### [ <processed1>, <processed2>, ... ]
    if cost != None:
        return (path, solution, depth, cost)
    else:
        return (path, solution, depth)

def bFs(source, destination, edges):
    """
    IN MY DESIGN
    EVERY NODE SAVES ITS PARENT'S INDEX
    IN ORDER TO DO BACKTRACKING..
    FOR EXAMPLE..
    NORMAL RESULT OF FIRST BFS EXAMPLE..
    [’izmir ’, ’manisa ’, ’usak ’, ’izmir ’, ’afyon ’, ’eskisehir ’, ’manisa ’, ’manisa ’, ’ankara ’]
    MY ACTUAL RESULT LIST
    [ [’izmir ’, -1], [’manisa ’, 0], [’usak ’, 1], [’izmir ’, 1], [’afyon ’, 1], [’eskisehir ’, 2], [’manisa ’, 2], [’manisa ’, 3], [’ankara ’, 4]]
    ALGORITHM IS ALMOST THE SAME TREE-SEARCH
    INSTEAD OF HOLDING SEPERATE QUEUE AND SOLUTION LIST, 
    I USED ONLY SOLUTION LIST AND WITH ITERATOR 'i' , I USED IT AS A QUEUE TOO.
    INCREASING 'i' MEANS DEQUEUE.
    """
    solution = []
    # -1 here implies that this node is root or source
    # it will be used when backtracking
    solution.append([source,-1])
    i = 0
    while True:
        # This means whether queue is empty...
        if i >= len(solution):
            return None
        
        # Seek front of queue
        leaf = solution[i] 
        
        # Test goal node
        if leaf[0] == destination: 
            break
        
        # Enqueue, explore neighbors, with parent's index.
        for e in edges:
            if e[0] == leaf[0]:
                solution.append([e[1], i])
            elif e[1] == leaf[0]:
                solution.append([e[0], i])
            else:
                continue
        
        # Pop queue
        i+=1

    return post_process(source,destination,solution,None)



def secondly(x):
    ### TO BE ABLE TO USE BUILT-IN SORT FUNCTION FOR HEAP PURPOSES...
    return x[2]

def uCs(source, destination, edges):
    solution = []

    ### IT IS SIMPLE LIST, I OFTEN SORT IT AND GET LEAST COSTLY ONE..
    ### structure is like [ <node>, <parent>, <total cost> ]
    heaplist = [[source,-1,0]]

    while len(heaplist)!=0:
        ### TAKE CLOSEST NODE
        leaf = heaplist.pop(0)
        solution.append(leaf)

        ## CHECK WHETHER IT IS THE GOAL NODE
        if leaf[0] == destination: 
            break
        
        ## EXPLORE NEW NODES
        for e in edges:
            if e[0] == leaf[0]:
                heaplist.append([e[1], len(solution)-1, leaf[2]+int(e[2])])
            elif e[1] == leaf[0]:
                heaplist.append([e[0], len(solution)-1, leaf[2]+int(e[2])])
            else:
                continue

        ## SORT EXPLORED NODES..
        heaplist.sort(key=secondly)
    
    return post_process(source,destination,solution,solution[-1][2])
    

def dLs(source, destination, edges, limit):
    solution = []
    """
    HERE MY NODE STRUCTURE ALMOST SAME, BUT
    DEPTH INFORMATION ADDED..
    [ <node name>, <parent index>, <depth> ]
    """
    stack = [ [source, -1, 0] ]
    while True:
        
        if len(stack)==0:
            return None
        leaf = stack.pop()
        solution.append(leaf)
        
        ## CHECK WHETHER IT IS THE GOAL NODE
        if leaf[0] == destination:
            break

        ## IF WE ARE AT LIMIT, GO BACK
        elif leaf[2] == limit:
            continue

        ## EXPLORE NEW NODES BY PUSHING THEM IN THE STACK..
        for e in edges:
            if e[0] == leaf[0]:
                stack.append([e[1], len(solution)-1, leaf[2]+1])
            elif e[1] == leaf[0]:
                stack.append([e[0], len(solution)-1, leaf[2]+1])
            else:
                continue      
    return post_process(source,destination,solution,None)

def iDf(source, destination, edges, limit): 
    ## It is just calling DLS by increasing
    ## limit from 0 to input limit 
    for i in range(limit+1):
        x = dLs(source,destination,edges,i)
        if x != None:
            # return with possibly low limit to achieve optimal solution
            return x
    return None

def UnInformedSearch (method_name, problem_file_name, maximum_depth_limit):
    ### READING INPUT...
    f = open(problem_file_name, "r")
    lines = f.readlines()
    f.close()
    source = lines[0][:-1]
    destination = lines[1][:-1]
    ### EDGE STRUCTURE IS LIKE
    ### [<node1>, <node2>, <distance>]
    edges = []
    for i in range(2,len(lines)):
        edges.append(lines[i].split(" "))
    ### EXECUTE DESIRED ALGORITHM...
    if method_name == "BFS":
        return bFs(source,destination,edges)
    elif method_name == "UCS":
        return uCs(source,destination,edges)
    elif method_name == "DLS":
        return dLs(source,destination,edges,maximum_depth_limit)
    elif method_name == "IDDFS":
        return iDf(source,destination,edges,maximum_depth_limit)
    return "Wrong method_name..."