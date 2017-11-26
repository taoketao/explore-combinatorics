# The point of this script is to help me develop on 
# issues that I still need to identify since it's been
# almost two months since I felt the code.
#
# Task: Run a simple curriculum learning experiment on 
# the flavor-place location task by Tse (2007): that is,
# with four starting locations, one of six signals, and
# six corresponding possible targets. Mimic 'digging'
# by placing a block over each starting location.

import sys

#from experiment import *
from Config import Config
from templates import *
from environment import *
from environment import environment_handler3

# Global sentinels:
X=0; Y=1;
MOVE_NORTH, MOVE_SOUTH, MOVE_EAST,  MOVE_WEST = 100,101,102,103
MOVE_FWD,   MOVE_BACK                         = 110,111
ROT_R90_C,  ROT_R180_C,  ROT_R270_C           = 120,121,122
DVECS = {MOVE_NORTH: (0,-1), MOVE_SOUTH: (0,1), MOVE_EAST: (1,0), \
        MOVE_WEST: (-1,0)}
DIRVECS = {(0,-1):'N', (0,1):'S', (1,0):'E', (-1,0):'W'}
OLAYERS = [agentLayer, goalLayer, immobileLayer, mobileLayer]

''' [Helper] Constants '''
N_ACTIONS = 4
XDIM=0; YDIM=0
agentLayer = 0
goalLayer = 1
immobileLayer = 2
mobileLayer = 3
N_EPS_PER_EPOCH = 4 # upper bound on number of initial start states there are

ALL = -1

''' [Default] Hyper parameters '''
TRAINING_EPISODES = 300;  # ... like an epoch
MAX_NUM_ACTIONS = 15;
EPSILON = 1.0;
REWARD = 1;
NO_REWARD = 0.0;
INVALID_REWARD = 0.0;
GAMMA = 0.9;
LEARNING_RATE = 0.01;
VAR_SCALE = 1.0; # scaling for variance of initialized values
INDICES_TO_CARDINAL_ACTIONS =\
        { 0:MOVE_NORTH, 1:MOVE_SOUTH, 2:MOVE_EAST, 3:MOVE_WEST }
# OLAYERS: ordered layers, where pos corresponds to value


AL,GL,IL,ML = [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]
AL[agentLayer]=1;       AL=np.array(AL)
GL[goalLayer]=1;        GL=np.array(GL)
ML[mobileLayer]=1;      ML=np.array(ML)
IL[immobileLayer]=1;    IL=np.array(IL)
not_a_layer = np.array([0,0,0,0])


# General utilities:
def map_nparr_to_tup(Iterable):
    return tuple([value.tolist()[0] for value in Iterable])

#def addvec(Iterable, m, optn=None):
#    try: 
#        return tuple([i+m for i,m in zip(Iterable,m)])
#    except:
#        return tuple([i+m for i in Iterable])
def addvec(Iterable, m, optn=None):
    try: 
        m[0]
    except:
        if m>80: 
            m = DVECS[m]
        else:
            m = DVECS[INDICES_TO_CARDINAL_ACTIONS[m]]
    return tuple([i+m for i,m in zip(Iterable,m)])

def multvec(Iterable, m, optn=None):
    if optn=='//':  return tuple([i//m for i in Iterable])
    if optn=='/':   return tuple([i/m for i in Iterable])
    if optn==int:   return tuple([int(i*m) for i in Iterable])
    return tuple([i*m for i in Iterable])

def at(mat, pos, lyr): return mat[pos[X], pos[Y], lyr]
def empty(mat, pos): return np.any(mat[pos[X], pos[Y], :])
def what(mat, pos): return np.array([at(mat, pos, lyr) for lyr in OLAYERS])
def put(mat, pos, lyr, v): mat[pos[X], pos[Y], lyr] = v
def put_all(mat, pos_list, lyr, v):
    for p in pos_list: put(mat, p, lyr, v)

#------#------#------#------#------#------#------#------#------#------#------#--
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#
#   Experiment API class:
#
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#------#------#------#------#------#------#------#------#------#------#------#--


# Experiment class:
class ExpAPI(environment_handler3):
    def __init__(self, experiment_name, centr, card_or_rot='card', debug=False):
        ''' Initializer for Experiment class. Please provide:
         - experiment_name keyword, which indirectly signals the starting 
            states. Currently takes 'tse2007'.
         - centr, the keyword for the reference frame. Currently takes 
            'allocentric' or 'egocentric' but in future will facilitate 
            rotational heading frames.
         - optional debug-mode boolean flag
        ''' 
        environment_handler3.__init__(self, gridsize = \
                { 'tse2007': (11,11), 'r-u': (7,7), 'ru': (7,7), \
                'r-u-ru': (7,7) }[experiment_name], \
                action_mode = centr, card_or_rot=card_or_rot   )
        self.centr = centr
        self.card_or_rot = card_or_rot
        self.state_gen = state_generator(self.gridsz)
        self.start_states = []
        self._set_starting_states({\
                'tse2007':TEMPLATE_TSE, \
                'r-u-ru': TEMPLATE_R_U_RU, \
                'r-u': TEMPLATE_R_U, \
                }[experiment_name], debug)
        self.experiment_name = experiment_name

    def _find_all(self, a_str, char):
        # [internal] scan from a template string (eg, TEMPLATE_TSE)
        s = a_str.replace(' ','')
        startX, startY = 0,0
        for c in s:
            if c==char: 
                yield((startX, startY))
            elif c=='r': 
                startY += 1
                startX = 0
            if c in 'a!xm.': 
                startX += 1

    # Set this experiment's possible starting states using complete template str
    def _set_starting_states(self, state_template, debug=False):
        oind = state_template.index('o')
        if state_template.index('e') > oind: raise Exception()
        num_start_locs = state_template.count('a')
        num_goal_locs = state_template.count('!')
        if not state_template.find('*') > oind: raise Exception()

        start_locs = list(self._find_all(state_template, 'a'))
        goal_locs = list(self._find_all(state_template, '!'));
        block_locs = list(self._find_all(state_template, 'x'));
        if 'D' in state_template:
            mobile_locs = list(self._find_all(state_template, '!'));
            self.valid_states = np.array( [AL, GL, AL|GL, IL, ML, ML|GL] ).T
        else:
            try:
                mobile_locs = list(self._find_all(state_template, 'm'));
            except:
                mobile_locs = []
                self.valid_states = np.array( [AL, GL, AL|GL, IL, ML] ).T
#        self.valid_states = np.append(self.valid_states, np.expand_dims(\
        #                np.array([0,0,0,0], dtype=bool)), axis=0)

        rx = [0,1,self.gridsz[X]-2, self.gridsz[X]-1]
        ry = [0,1,self.gridsz[Y]-2, self.gridsz[Y]-1]

        ''' flavor == goal here. '''
        for start_box in start_locs:
            for flav_id, flavor_loc in enumerate(goal_locs):
                st = np.zeros( (self.gridsz[X], self.gridsz[Y], NUM_LAYERS))
                put(st, start_box, agentLayer, True)
                put(st, flavor_loc, goalLayer, True)
                put_all(st, mobile_locs, mobileLayer, True)
                put_all(st, block_locs,  immobileLayer, True)

                self.start_states.append( { 'flavor signal': flav_id, \
                        'state': st, '_whichgoal':flav_id, \
                        '_startpos':start_box, 'goal loc':flavor_loc })
                #        rnd_state = self.start_states[np.random.choice(range(24))]


        self.curr_sorted_states = self.start_states.copy()
        def dist(state_):
            x,y = state_['goal loc'], state_['_startpos']
            return abs(x[0]-y[0])+abs(x[1]-y[1])
        self.curr_sorted_states.sort(key=dist)


        rnd_state = np.random.choice(self.start_states)
        if debug: 
            print('flag 93747')
            print_state(rnd_state, 'condensed')

    def _view_state_copy(self, st):
        sret = {}
        for key in ('_startpos','flavor signal','_whichgoal'):
            sret[key] = st[key]
        sret['state'] = np.copy(st['state'])
        return sret

    def get_random_starting_state(self): 
        ''' Public method: get a random state struct with fields: 'state', 
        '_startpos', 'flavor signal', '_whichgoal', 'goal loc'. The three 
        later fields are helper attributes for, say, curricula or presentation.   
        '''
        #st = self.start_states[np.random.choice(range(24))]
        return self._view_state_copy(np.random.choice(self.start_states))


    def get_weighted_starting_state(self, envir, pct):
        pct=float(pct)
#        print(envir,pct, 0.5*pct,1-pct)
        if envir=='r-u': raise Exception()
        assert (envir=='r-u-ru')
        ps = [0.5*pct, 0.5*pct, 1-pct]
        return self._view_state_copy(np.random.choice(\
                self.curr_sorted_states, p=ps))

#        for s in self.start_states:
#            print([s[x] for x in ['goal loc','_startpas'] ])
#        for s in self.curr_sorted_states:
#            print([s[x] for x in ['goal loc','_startpos'] ])


    def get_starting_state(self, curriculum_name, epoch, envir=None): 
        # interface wrapper method for submethods
        curr = curriculum_name
        cspl = curriculum_name.split(':')
        if curr==None:   return self.get_random_starting_state()['state']
        elif len(curr)>4 and curr[:4]=='FLAT' and len(cspl)==2:
            return self.get_weighted_starting_state(envir, float(cspl[1]))['state']
        elif len(curr)>6 and curr[:6]=='STEP-1' and len(cspl)==4:     
            if epoch >= int(cspl[3]):
                return self.get_weighted_starting_state(envir, cspl[2])['state']
            else:
                return self.get_weighted_starting_state(envir, cspl[1])['state']
        elif len(curr)>8 and curr[:8]=='LINEAR-1' and len(cspl)==4:
            param = min(1.0, max(0.0, epoch/float(cspl[3])))
            pct = param*float(cspl[2])+(1-param)*float(cspl[1])
            return self.get_weighted_starting_state(envir, pct)['state']
        else:
            raise exception(curr, cspl, epoch, envir) 

        return curriculum_name, 'error expenv line ~200'

        # Very hacky:
        assert(self.experiment_name=='r-u-ru')
        l1 = len(TEMPLATE_R_U)
        l2 = len(TEMPLATE_RU)
        if curriculum_name=='STEP':
            ps = [0.5, 0.5, 0] if False else False
        return self._view_state_copy(np.random.choice(self.start_states), p=ps)['state']

    def get_all_starting_states(self):
        ''' Public method: get a random state struct with fields: 'state', 
        '_startpos', 'flavor signal', '_whichgoal', 'goal loc'. The three 
        later fields are helper attributes for, say, curricula or presentation.   
        '''
        return [self._view_state_copy(st) for st in self.start_states]

    def get_agent_loc(self,state):    
        '''Public method: query the location of the agent. (<0,0> is NW corner.)'''
        return self._get_loc(state,targ='agent')

    def get_goal_loc(self,s):     return self._get_loc(s,targ='goal')
#    def get_allo_loc(self,s):     return self._get_loc(s,targ='map') # center

    def _get_loc(self, state_matrix, targ):
        if targ=='agent': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,agentLayer]==1))
        if targ=='goal': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,goalLayer]==1))
        if targ=='map':
            return multvec(self.gridsz, 2, '//') # center

    def _out_of_bounds(self, pos):
        return (pos[X] < 0 or pos[X] >= self.gridsz[X] or \
                pos[Y] < 0 or pos[Y] >= self.gridsz[Y])

    def _is_valid_move(self, st, move): 
        aloc = self.get_agent_loc(st)
        newaloc = addvec(aloc, move)
        if self._out_of_bounds(newaloc): return False
        if at(st, newaloc, immobileLayer): return False
        if at(st, newaloc, mobileLayer):
            st2 = np.copy(st)
            put(st2, newaloc, agentLayer, True)
            put(st2, aloc, agentLayer, False)
            return self._is_valid_move(st2, move)
        return True

    def _move_ent_from_to(self, mat, loc, nextloc, lyr):
        m2 = np.copy(mat)
        if not at(m2,loc,lyr): raise Exception()
        #print ("Adjusting",lyr,loc,nextloc)
        put(m2,loc,lyr, False)
        put(m2,nextloc,lyr, True)
        return m2

    def _adjust_blocks(self, mat, loc, dir_vec, debug=True):
        nloc = addvec(loc, dir_vec)
        if self._out_of_bounds(nloc): return mat, False
        arr = [what(mat, loc), what(mat, nloc)]
        ploc=nloc
        while True:
            nloc = addvec(ploc, dir_vec)
            #print('>>',dir_vec)
            if self._out_of_bounds(nloc): return mat, False
            if not arr[-1][mobileLayer]: return mat, not arr[-1][immobileLayer]
            nmat = self._move_ent_from_to(mat, ploc, nloc, mobileLayer)
            if len(arr)>2: put(nmat, ploc, mobileLayer, True)
            arr.append(what(mat, nloc))
            ploc=nloc
            mat=nmat
        raise Exception()

    def _move_agent(self, state_mat, dir_vec, ret_valid_move):
        aloc = self.get_agent_loc(state_mat)
        newL = addvec(aloc, dir_vec)
        state_mat2, success = self._adjust_blocks(state_mat, aloc, dir_vec)
        state_mat2 = self._move_ent_from_to(state_mat2, aloc, newL, agentLayer)
        if self.centr == 'egocentric':
            shft, axis = { 0:(1,1), 1:(-1,1), 2:(-1,0), 3:(1,0) }[dir_vec]
            state_mat2=np.roll(state_mat2, shift=shft, axis=axis)
        elif not self.centr == 'allocentric': raise Exception(self.centr)

        isValid = self._is_valid_move(state_mat, dir_vec)
        if isValid: 
            if ret_valid_move==True: return state_mat2, isValid
            else: return state_mat2
        if ret_valid_move==True: return state_mat, isValid
        return state_mat

    def _rot_agent(self, state_mat, nrots, ret_valid_move):
        aloc = self.get_agent_loc(state_mat)
        assert(self.experiment_name == 'r-u-ru')
        assert(nrots in [1,2,3])
        state_mat = np.rot90(state_mat, k=nrots, axes=(0,1))
#        state_mat2=np.roll(state_mat2, shift=shft, axis=axis)
        if self.centr == 'egocentric':
            centr_pos = (3,3)
            dx, dy = aloc[0]-centr_pos[0], aloc[1]-centr_pos[1]
            state_mat = np.roll(state_mat, shift=dx, axis=0)
            state_mat = np.roll(state_mat, shift=dy, axis=1)
        elif not self.centr == 'allocentric': raise Exception(self.centr)
        return state_mat, True


    def new_statem(self, orig_state, action, valid_move_too=False):
        '''Public Method: error-proofed public method for making (S') from (S,A)
        NOT currently errorproofed against egocentrism!'''
        if action>=100:
            return self._move_agent(orig_state, DVECS[action], valid_move_too)
        else:
            if self.card_or_rot=='card':
                return self._move_agent(orig_state, action, valid_move_too)
            elif self.card_or_rot=='rot':
                if action==0:
                    return self._move_agent(orig_state, 0, valid_move_too)
                else:
                    return self._rot_agent(orig_state, action, valid_move_too)



def _____dont_do_this__stub():
    for centr in ['egocentric', 'allocentric']:
        ExpAPI('tse2007', centr)._set_starting_states(TEMPLATE_TSE)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# fun little test script:
if __name__=='__main__':
    ex = ExpAPI('tse2007', 'egocentric')
    cur_state = ex.get_random_starting_state()['state']
    while False:#True:
        print('current state:') 
        print('flag 36351')
        print_state(cur_state, 'condensed')
        print('current location:', ex.get_agent_loc(cur_state))
        inp = input(' interface input >> ')
        if not len(inp)==1: break
        try:
            inp_to_mov = {\
                    'N': MOVE_NORTH,
                    'S': MOVE_SOUTH,
                    'E': MOVE_EAST,
                    'W': MOVE_WEST, }[inp.upper()]
        except: 
            break
        next_state = ex.new_statem(cur_state, inp_to_mov)

        cur_state = next_state
