def fourthly(x):
    ### TO BE ABLE TO USE BUILT-IN SORT FUNCTION FOR HEAP PURPOSES...
    return x[3]

def heuristic_maze(state, goal, isAStar):
    # THIS HEURISTIC FUNCTION FOR MAZE USES 'MANHATTAN DISTANCE' FROM STATE LOCATION TO GOAL LOCATION..
    # DISTANCE = DIFFERENCE BTW Y-AXIS OF CURRENT LOCATION AND GOAL LOCATION + DIFFERENCE BTW X-AXIS OF CURRENT LOCATION AND GOAL LOCATION 
    return abs(state[0]-goal[0]) + abs(state[1]-goal[1]) if isAStar==True else 0

def MAZE(method_name, problem_file_name):
    # READING INPUT...
    f = open(problem_file_name, "r")
    lines = f.readlines()
    f.close()
    """
    Because of nested access list order
    I use location like (y-axis, x-axis)
    contrary to HW2.pdf, however I fix it
    in post process
    """
    start, goal = eval(lines[0]), eval(lines[1])
    start, goal = (start[1],start[0]), (goal[1],goal[0])
    grid = lines[2:]
    width, height = len(grid[0])-1, len(grid)

    # IF START STATE IS ALREADY OBSTACLE...
    if grid[start[0]][start[1]] == "#":
        return None
    
    """
    My node structure is like below
    [ <state>, <parent_index_in_solution>, <last_move>, <heuristic>, <total_cost(depth)>]
    """
    start = [start, -1, None, -1, 0]
    solution = []
    explored = [] ## This just hold locations of explored states.
    heaplist = [start]  # [ [(0,0), -1, ...] ]
    isAStar = True if method_name == "AStar" else False
    while True:
        if len(heaplist)==0:
            return None
        leaf = heaplist.pop(0)
        if leaf[0] in explored:
            continue
        solution.append(leaf)
        if leaf[0] == goal:
            break
        explored.append(leaf[0])
        y = leaf[0][0]
        x = leaf[0][1]
        """
        In this EXPLORATION part, it is important not to add heuristic to cumulative cost.
        Therefore, I save total cost at leaf[4] and heuristic+total_cost, which decides the move, at leaf[3].
        ORDER FOR ADDING TO HEAP IS LEFT-UP-RIGHT-DOWN AS MENTIONED IN HW2.PDF
        IF ALL COSTS ARE 1, WE POP FROM HEAP LIKE NORMAL QUEUE AND HEAP IS NOT AFFECTED BY SORT FUNCTION
        """
        # left 
        if x > 0 and grid[y][x-1]!='#' and (y,x-1) not in explored:
            n4 = (y,x-1)
            heaplist.append([n4, len(solution)-1, "LEFT", heuristic_maze(n4,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # up
        if y > 0 and grid[y-1][x]!='#' and (y-1,x) not in explored:
            n2 = (y-1,x)
            heaplist.append([n2, len(solution)-1, "UP", heuristic_maze(n2,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # right
        if x < width-1 and grid[y][x+1]!='#' and (y,x+1) not in explored:
            n3 = (y,x+1)
            heaplist.append([n3, len(solution)-1, "RIGHT", heuristic_maze(n3,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # down
        if y < height-1 and grid[y+1][x]!='#' and (y+1,x) not in explored:
            n1 = (y+1,x) 
            heaplist.append([n1, len(solution)-1, "DOWN", heuristic_maze(n1,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        heaplist.sort(key=fourthly)

    """
    solution list now is like below
    [ processed1, processed2, ..., processedx (destination)]
    Now, I need to extract desired information from my raw data
    Let's do backtracking by using saved parent index data
    """
    path = []
    depth = 0
    i = len(solution)-1
    while True:
        path = [solution[i][0]] + path
        i = solution[i][1]
        if i == -1:
            break
        depth += 1
    # convert locations (y,x) into desired (x,y)
    path = [(p[1],p[0]) for p in path]
    # remove unnecesary attributes
    solution = [(s[0][1],s[0][0]) for s in solution]
    return (path,solution,depth,depth)

def deepCopy(leaf):
    # It is simply copying a 3x3 puzzle grid.
    return [[leaf[0][0], leaf[0][1], leaf[0][2]], [leaf[1][0], leaf[1][1], leaf[1][2]], [leaf[2][0], leaf[2][1], leaf[2][2]]]

def makeStr(leaf):
    # This function converts a puzzle grid into desired string representation.
    strink = ""
    for i in range(3):
        for j in range(3):
            if leaf[i][j]==0:
                strink = strink + ' '
            else:
                strink = strink + str(leaf[i][j])
    return strink

def heuristic_puzzle(state, goal, isAStar):
    # THIS HEURISTIC FUNCTION FOR EIGHTPUZZLE USES 'MANHATTAN DISTANCE' FROM STATE LOCATION TO GOAL LOCATION..
    if isAStar==False:
        return 0
    heu = 0
    a,b,c,d=0,0,0,0
    for x in range(1,9):
        for i in range(3):
            for j in range(3):
                if state[i][j] == x:
                    a,b = i,j
        for i in range(3):
            for j in range(3):
                if goal[i][j] == x:
                    c,d = i,j
        heu += abs(a-c) + abs(b-d)
    return heu
    

def PUZZLE(method_name, problem_file_name):
    # READING INPUT...
    f = open(problem_file_name, "r")
    lines = f.readlines()
    f.close()
    row0, row1, row2 = [int(x) for x in lines[0].split(' ')], [int(x) for x in lines[1].split(' ')], [int(x) for x in lines[2].split(' ')]
    start = [row0, row1, row2]
    row0, row1, row2 = [int(x) for x in lines[4].split(' ')], [int(x) for x in lines[5].split(' ')], [int(x) for x in lines[6].split(' ')]
    goal = [row0, row1, row2]
    
    """
    My node structure is like below
    [ <state>, <parent_index_in_solution>, <last_move>, <heuristic>, <total_cost(depth)>]
    """
    explored = []
    start = [start, -1, None, 0, 0]
    solution = []
    heaplist = [start]
    isAStar = True if method_name == "AStar" else False
    y, x = 0, 0
    while True:
        if len(heaplist)==0:
            return None
        leaf = heaplist.pop(0)
        if leaf[0] in explored:
            continue
        solution.append(leaf)
        if leaf[0] == goal:
            break
        explored.append(leaf[0])
        # Find currently empty indices
        for k in range(3):
            for j in range(3):
                if leaf[0][k][j] == 0:
                    y, x = k, j
                    break
        """
        In this EXPLORATION part, it is important not to add heuristic to cumulative cost.
        Therefore, I save total cost at leaf[4] and heuristic+total_cost, which decides the move, at leaf[3].
        ORDER FOR ADDING TO HEAP IS LEFT-UP-RIGHT-DOWN AS MENTIONED IN HW2.PDF
        IF ALL COSTS ARE 1, WE POP FROM HEAP LIKE NORMAL QUEUE AND HEAP IS NOT AFFECTED BY SORT FUNCTION.
        """
        # left 
        if x > 0:
            n4 = deepCopy(leaf[0])
            n4[y][x] = n4[y][x-1]
            n4[y][x-1] = 0 
            heaplist.append([n4, len(solution)-1, "LEFT", heuristic_puzzle(n4,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # up
        if y > 0:
            n2 = deepCopy(leaf[0])
            n2[y][x] = n2[y-1][x]
            n2[y-1][x] = 0 
            heaplist.append([n2, len(solution)-1, "UP", heuristic_puzzle(n2,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # right
        if x < 2:
            n3 = deepCopy(leaf[0])
            n3[y][x] = n3[y][x+1]
            n3[y][x+1] = 0
            heaplist.append([n3, len(solution)-1, "RIGHT", heuristic_puzzle(n3,goal,isAStar)+leaf[4]+1, leaf[4]+1])
        # down
        if y < 2:
            n1 = deepCopy(leaf[0])
            n1[y][x] = n1[y+1][x]
            n1[y+1][x] = 0 
            heaplist.append([n1, len(solution)-1, "DOWN", heuristic_puzzle(n1,goal,isAStar)+leaf[4]+1, leaf[4]+1]) 

        heaplist.sort(key=fourthly)

    """
    solution list now is like below
    [ processed1, processed2, ..., processedx (destination)]
    Now, I need to extract desired information from my raw data
    Let's do backtracking by using saved parent index data
    """
    path = []
    strinks = []
    depth = 0
    i = len(solution)-1
    while True:
        path = [solution[i][2]] + path
        strinks.append(makeStr(solution[i][0]))
        i = solution[i][1]
        if i == -1:
            break
        depth += 1
    
    return (path[1:],strinks[::-1],depth,depth)

def InformedSearch(method_name, problem_file_name):
    if problem_file_name[:4] == "maze":
        return MAZE(method_name,problem_file_name)
    elif problem_file_name[:11] == "eightpuzzle":
        return PUZZLE(method_name,problem_file_name)
    else:
        return "Wrong file name..."