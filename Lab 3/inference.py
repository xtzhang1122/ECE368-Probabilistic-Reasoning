import numpy as np
import graphics
import rover
import collections

def log_function(s):
    if s == 0:
        return -np.inf
    else:
        return np.log(s)

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [prior_distribution]
    backward_messages = [None] * num_time_steps
    marginals = [] 
    
    # forward messages computation
    for i in range(num_time_steps - 1):
        if observations[i] != None:
            prob_y_given = rover.Distribution()

            for state, probability in forward_messages[i].items():
                prob_y_given[state] = probability * observation_model(state)[observations[i]]

            prob_y_given.renormalize()

        else:
            prob_y_given = forward_messages[i]
        
        next_step = rover.Distribution()
        for state, probability in prob_y_given.items():
            for next_state, transition_prob in transition_model(state).items():
                next_step[next_state] += probability * transition_prob
        
        next_step.renormalize()
        forward_messages.append(next_step)

    # backward messages computation
    backward_transition_dic = collections.defaultdict(rover.Distribution) # defaultdict(<class 'rover.Distribution'>, {})
    
    for i in all_possible_hidden_states: # i is a state
        for state, probability in transition_model(i).items():
            backward_transition_dic[state][i] += probability

    uniform = rover.Distribution()
    for state in all_possible_hidden_states:
        uniform[state] = 1
    uniform.renormalize()

    backward_messages[-1] = uniform
    for i in range(num_time_steps - 1, 0, -1):
        if observations[i] != None:
            prob_y_given = rover.Distribution()
            for state, probability in backward_messages[i].items():
                prob_y_given[state] = probability * observation_model(state)[observations[i]]
            prob_y_given.renormalize()
        
        else:
            prob_y_given = backward_messages[i]
    
        prev_state = rover.Distribution()
        for state, probability in prob_y_given.items():
            for past_state, transition_prob in backward_transition_dic[state].items():
                prev_state[past_state] += probability * transition_prob
        
        prev_state.renormalize()
        backward_messages[i - 1] = prev_state

    # compute the marginals 
    for i in range(num_time_steps):
        m = rover.Distribution()

        for state in all_possible_hidden_states: 
            joint = forward_messages[i][state] * backward_messages[i][state]
            if observations[i] != None:
                joint *= observation_model(state)[observations[i]]

            if joint > 0:
                m[state] = joint
        
        m.renormalize()
        marginals.append(m)
    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    forward = []
    initial_map = {}

    for state in all_possible_hidden_states:
        initial_map[state] = [log_function(prior_distribution[state]), None]

        if observations[0] != None:
            initial_map[state][0] += log_function(observation_model(state)[observations[0]])

    forward.append(initial_map)

    for i in range(1, num_time_steps):
        new_map = collections.defaultdict(lambda: (-np.inf, None))

        for old_state, (old_probability, _) in forward[-1].items():
            for new_state, transition_probability in transition_model(old_state).items():
                total_probability = old_probability + log_function(transition_probability)

                if total_probability > new_map[new_state][0]:
                    new_map[new_state] = (total_probability, old_state)

        if observations[i] != None:
            for state in all_possible_hidden_states:
                log_probability, back = new_map[state]
                new_map[state] = (log_probability + log_function(observation_model(state)[observations[i]]), back)
        
        forward.append(new_map)

    estimated_hidden_states[num_time_steps - 1] = max(forward[num_time_steps - 1], key=lambda s: forward[num_time_steps - 1][s][0])

    for i in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[i] = forward[i + 1][estimated_hidden_states[i + 1]][1]
  
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])

    temp = []
    for i in range(timestep):
        temp.append(sorted(marginals[i].items(), key=lambda x: x[1], reverse=True)[0])
    
    print('\n')


    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    error1 = 0 # for Question 2
    error2 = 0 # for Question 3
    for i in range(timestep):
        found = False
        for entry in temp[i]:
            if hidden_states[i] == entry:
                found = True
        if found == False:
            error1 += 1

        if hidden_states[i] != estimated_states[i]:
            error2 += 1

    print('Error Probability for Question 2')
    print(error1 / 100)
    print('Error Probability for Question 3')
    print(error2 / 100)

    print('\n')
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    print('\n')
    # for Question 5
    # print("All hidden states in the MAP estimate:")
    # for time_step in range(num_time_steps):
    #     print(time_step, estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
