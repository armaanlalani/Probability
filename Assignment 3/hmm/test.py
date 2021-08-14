import numpy as np
import graphics
import rover

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
    forward_messages = [None] * num_time_steps
    #forward_messages[0] = prior_distribution 
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    # initialization of forward message
    forward_messages[0] = rover.Distribution({})
    initial_observed_position = observations[0]
    for z0 in all_possible_hidden_states:
        if initial_observed_position == None:
            initial_prob_position_on_state = 1
        else:
            initial_prob_position_on_state = observation_model(z0)[initial_observed_position]
        prior_z0 = prior_distribution[z0]
        if (initial_prob_position_on_state * prior_z0) != 0:
            forward_messages[0][z0] = initial_prob_position_on_state * prior_z0
    forward_messages[0].renormalize()
    
    # when i >= 1
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution({})
        observed_position = observations[i]
        for zi in all_possible_hidden_states:
            if observed_position == None:
                prob_position_on_state = 1
            else:               
                prob_position_on_state = observation_model(zi)[observed_position]
            
            sum = 0
            for zi_minus_1 in forward_messages[i-1]:
                sum = sum + forward_messages[i-1][zi_minus_1] * transition_model(zi_minus_1)[zi]
            if (prob_position_on_state * sum) != 0: # only save non-zero values
                forward_messages[i][zi] = prob_position_on_state * sum

        forward_messages[i].renormalize() # normalize forward messages
    #print(forward_messages)
    
    # TODO: Compute the backward messages
    # initialization of backward message
    backward_messages[num_time_steps-1] = rover.Distribution({})
    for zn_minus_1 in all_possible_hidden_states:
        backward_messages[num_time_steps-1][zn_minus_1] = 1
    # when backward message is not the last one
    for i in range(1, num_time_steps):
        backward_messages[num_time_steps-1-i] = rover.Distribution({})
        for zi in all_possible_hidden_states:
            sum = 0
            for zi_plus_1 in backward_messages[num_time_steps-1-i+1]:
                observed_position = observations[num_time_steps-1-i+1]
                if observed_position == None:
                    prob_position_on_next_state = 1
                else:
                    prob_position_on_next_state = observation_model(zi_plus_1)[observed_position]
                sum = sum + backward_messages[num_time_steps-1-i+1][zi_plus_1] * prob_position_on_next_state * transition_model(zi)[zi_plus_1]
            if sum != 0:
                backward_messages[num_time_steps-1-i][zi] = sum
        backward_messages[num_time_steps-1-i].renormalize()
    
    print(backward_messages)

    # TODO: Compute the marginals
    for i in range (0, num_time_steps): 
        marginals[i] = rover.Distribution({})    
        sum = 0
        for zi in all_possible_hidden_states:
            if forward_messages[i][zi] * backward_messages[i][zi] != 0:
                marginals[i][zi] = forward_messages[i][zi] * backward_messages[i][zi]
                sum = sum + forward_messages[i][zi] * backward_messages[i][zi]
        for zi in marginals[i].keys():
            marginals[i][zi] = marginals[i][zi] / sum

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

    # TODO: Write your code here   
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    z_previous = [None] * num_time_steps

    # initialization
    w[0] = rover.Distribution({})
    initial_observed_position = observations[0]
    for z0 in all_possible_hidden_states:
        if initial_observed_position == None:
            initial_prob_position_on_state = 1
        else:
            initial_prob_position_on_state = observation_model(z0)[initial_observed_position]
        prior_z0 = prior_distribution[z0]
        if (initial_prob_position_on_state != 0) and (prior_z0 != 0):
            w[0][z0] = np.log(initial_prob_position_on_state) + np.log(prior_z0)
    
    # when i >= 1
    for i in range(1, num_time_steps):
        print(i)
        w[i] = rover.Distribution({})
        z_previous[i] = dict()
        observed_position = observations[i]
        for zi in all_possible_hidden_states:
            if observed_position == None:
                prob_position_on_state = 1
            else:
                prob_position_on_state = observation_model(zi)[observed_position]
            max_term = -np.inf
            for zi_minus_1 in w[i-1]:
                if transition_model(zi_minus_1)[zi] != 0:
                    print(transition_model(zi_minus_1)[zi], prob_position_on_state)
                    potential_max_term = np.log(transition_model(zi_minus_1)[zi]) + w[i-1][zi_minus_1]
                    if (potential_max_term > max_term) and (prob_position_on_state != 0):
                        max_term = potential_max_term
                        z_previous[i][zi] = zi_minus_1 # keep track of which zi_minus_1 can maximize w[i][zi]


            if prob_position_on_state != 0:
                w[i][zi] = np.log(prob_position_on_state) + max_term
            
    # back track to find z0 to zn
    # first, find zn* (the last)
    max_w = -np.inf
    for zi in w[num_time_steps-1]:
        potential_max_w = w[num_time_steps-1][zi]
        if potential_max_w > max_w:
            max_w = potential_max_w
            estimated_hidden_states[num_time_steps-1] = zi
    print(z_previous)
    for i in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps-1-i] = z_previous[num_time_steps-i][estimated_hidden_states[num_time_steps-i]]

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


   
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # error calculation
    correct_v = 0
    for i in range(0, num_time_steps):
        if hidden_states[i] == estimated_states[i]:
            correct_v = correct_v + 1
    print("viterbi's error is:", 1-correct_v/100)

    correct_fb = 0
    for i in range(0, num_time_steps):
        predict_z = None
        max_prob = 0
        for zi in marginals[i]:
            if marginals[i][zi] > max_prob:
                predict_z = zi
                max_prob = marginals[i][zi]
        print(i, ":", predict_z)
        if hidden_states[i] == predict_z:
            correct_fb = correct_fb + 1
    print("forward & backward's error is:", 1-correct_fb/100)

    
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