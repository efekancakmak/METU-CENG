import random

#Student name & surname: Efekan Çakmak
#Student ID: 2309797

## hocam, trustly, I started this homework last two days
## First day writed all lines
## but can not debug my code whole second day.
## in some cases my code gives approximate results, sometimes diverges.

def SolveMDP (method_name , problem_file_name, random_seed):
    random.seed(random_seed)
    # Reading input file..
    f = open(problem_file_name,'r')
    l = f.readlines()
    f.close()
    environment = (eval(l[1].split(' ')[0]),eval(l[1].split(' ')[1]))
    obstacles = [eval(i) for i in l[3].split('|')]
    goals = eval('{'+ l[5].replace('|',',') + '}')
    start = eval(l[7])
    reward = eval(l[9])
    action_noise = [eval(l[11]), eval(l[12]), eval(l[13])]
    learning_rate = eval(l[15])
    gamma = eval(l[17])
    epsilon = eval(l[19])
    episode_count = eval(l[21])

    actions = ['<' , '^' , '>' , 'V']
    if method_name == "TD(0)":
        utilities = {}
        pol = {}
        allgrid = {} # to check whether state is outside
        for i in range(environment[0]):
            for j in range(environment[1]):
                allgrid[(i,j)] = 'dummy'
                utilities[(i,j)] = 0
        for i in range(episode_count):
            state = start
            while state not in goals:
                if random.random() <= epsilon:
                    action = actions[random.randint(0, 3)]
                else:
                    l = u = r = d = -99**9
                    if (state[0],state[1]-1) in allgrid:
                        l = utilities[(state[0],state[1]-1)]
                    else:
                        l = utilities[(state[0],state[1])]
                    if (state[0]-1, state[1]) in allgrid:
                        u = utilities[(state[0]-1, state[1])]
                    else:
                        u = utilities[(state[0],state[1])]
                    if (state[0], state[1]+1) in allgrid:
                        r = utilities[(state[0], state[1]+1)]
                    else:
                        r = utilities[(state[0],state[1])]
                    if (state[0]+1, state[1]) in allgrid:
                        d = utilities[(state[0]+1, state[1])]
                    else:
                        d = utilities[(state[0],state[1])]
                    
                    m = max([l,u,r,d])
                    if m == l:
                        action = '<' 
                    elif m == u:
                        action = '^' 
                    elif m == r:
                        action = '>' 
                    elif m == d:
                        action = 'V'

                # apply action noises...
                if action == '<':
                    action = random.choices(['V', '<' ,'^'] , weights=[action_noise[1] , action_noise[0] , action_noise[2]])[0]
                elif action == '^':
                    action = random.choices(['<' ,'^', '>'] , weights=[action_noise[1] , action_noise[0] , action_noise[2]])[0]
                elif action == '>':
                    action = random.choices(['^', '>', 'V'] , weights=[action_noise[1] , action_noise[0] , action_noise[2]])[0]
                else:
                    action = random.choices(['>', 'V', '<'] , weights=[action_noise[1] , action_noise[0] , action_noise[2]])[0]

                # resulted actions
                if action == '<':
                    next_state = (state[0], state[1]-1)
                elif action == '^':
                    next_state = (state[0]-1, state[1])
                elif action == '>':
                    next_state = (state[0], state[1]+1)
                else:
                    next_state = (state[0]+1, state[1])
                

                pol[state] = action
                
                ## state is outside or an obstacle
                if (next_state not in allgrid) or (next_state in obstacles):
                    next_state = (state[0],state[1])
                    utilities[state] = utilities[state] + learning_rate*(reward + gamma*utilities[next_state] - utilities[state])
                    continue
                elif next_state in goals:
                    r = reward + goals[next_state]
                    utilities[state] = utilities[state] + learning_rate*(r + gamma*utilities[next_state] - utilities[state])
                    break
                
                utilities[state] = utilities[state] + learning_rate*(reward + gamma*utilities[next_state] - utilities[state])

                state = (next_state[0],next_state[1])

        for g in goals:
            utilities[g] = goals[g]
        for o in obstacles:
            utilities.pop(o)
        for u in utilities:
            utilities[u] = round(utilities[u],2)
        return utilities,pol
        
    elif method_name == "Q-learning":
        Qtable = {}
        allgrid = {} # to check whether state is outside
        for i in range(environment[0]):
            for j in range(environment[1]):
                allgrid[(i,j)] = 'dummy'
                for a in actions:
                    Qtable[(i,j),a] = 0
        
        for i in range(episode_count):
            state = (start[0],start[1])
            while state not in goals:
                if random.random() <= epsilon:
                    action = random.choice(actions)
                else:
                    max_q = -99**9
                    for a in actions:
                        if Qtable[(state,a)] > max_q:
                            action = a
                            max_q = Qtable[(state,a)]
                
                # apply action noises...
                if action == '<':
                    action = random.choices(['V', '<' ,'^'], weights = [action_noise[1], action_noise[0], action_noise[2]])[0]
                elif action == '^':
                    action = random.choices(['<' ,'^', '>'], weights = [action_noise[1], action_noise[0], action_noise[2]])[0]
                elif action == '>':
                    action = random.choices(['^', '>', 'V'], weights = [action_noise[1], action_noise[0], action_noise[2]])[0]
                else:
                    action = random.choices(['>', 'V', '<'], weights = [action_noise[1], action_noise[0], action_noise[2]])[0]

                # resulted actions
                if action == '<':
                    next_state = (state[0], state[1]-1)
                elif action == '^':
                    next_state = (state[0]-1, state[1])
                elif action == '>':
                    next_state = (state[0], state[1]+1)
                else:
                    next_state = (state[0]+1, state[1])
                
                ## whether state is outside or an obstacle
                if next_state not in allgrid or next_state in obstacles:
                    next_state = (state[0],state[1])
                    
                if next_state in goals:
                    r = reward + goals[next_state]
                    Qtable[(state,action)] = Qtable[(state,action)] + learning_rate*(r - Qtable[(state,action)])
                    break

                max_q = -99**9
                for a in actions:
                    max_q = max(max_q,Qtable[next_state,a])
                Qtable[(state,action)] = Qtable[(state,action)] + learning_rate*(reward + gamma*max_q - Qtable[(state,action)])

                state = (next_state[0],next_state[1])

        # policy extract..
        utils = {}
        pol = {}
        for i in range(environment[0]):
            for j in range(environment[1]):
                if (i,j) in goals or (i,j) in obstacles:
                    continue
                max_q = -99**9
                for a in actions:
                    if Qtable[(i,j),a] > max_q:
                        action = a
                        max_q = Qtable[(i,j),a]
                utils[i,j] = max_q
                pol[i,j] = action

        return utils, pol
    else:
        print("Wrong method name!")
        exit()

#print(SolveMDP("TD(0)",'io/mdp1.txt',37))
#print(SolveMDP("Q-learning",'io/mdp1.txt',462))