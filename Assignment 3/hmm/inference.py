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
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    transition_m = rover.Distribution()
    observation_m = rover.Distribution()
    for hidden in all_possible_hidden_states:
        transition_m[hidden] = transition_model(hidden)
        observation_m[hidden] = observation_model(hidden)

    # TODO: Compute the forward messages
    for time in range(0,num_time_steps):
        forward_messages[time] = rover.Distribution()
        for z in all_possible_hidden_states:
            if observations[time] == None:
                prob = 1
            else:
                prob = observation_m[z][observations[time]]
            if time == 0 and prob * prior_distribution[z] != 0:
                forward_messages[time][z] = prob * prior_distribution[z]
            elif time != 0:
                sigma = 0
                for zn in forward_messages[time-1]:
                    sigma += forward_messages[time-1][zn] * transition_m[zn][z]
                if prob * sigma != 0:
                    forward_messages[time][z] = prob * sigma
        forward_messages[time].renormalize()
                   
    # TODO: Compute the backward messages
    for time in range(num_time_steps-1,-1,-1):
        backward_messages[time] = rover.Distribution()
        if time == num_time_steps-1:
            for state in all_possible_hidden_states:
                backward_messages[time][state] = 1
            continue
        for z in all_possible_hidden_states:
            sigma = 0
            for zn in backward_messages[time+1]:
                if observations[time+1] == None:
                    prob = 1
                else:
                    prob = observation_m[zn][observations[time+1]]
                sigma += backward_messages[time+1][zn] * prob * transition_m[z][zn]
            if sigma != 0:
                backward_messages[time][z] = sigma
        backward_messages[time].renormalize()

    # TODO: Compute the marginals 
    for time in range(num_time_steps):
        marginals[time] = rover.Distribution()
        for z in all_possible_hidden_states:
            if forward_messages[time][z] * backward_messages[time][z] != 0:
                marginals[time][z] = forward_messages[time][z] * backward_messages[time][z]
        sigma = sum(marginals[time].values())
        for z in marginals[time].keys():
            marginals[time][z] /= sigma
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
    w = [None] * num_time_steps
    phi = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    transition_m = rover.Distribution()
    observation_m = rover.Distribution()
    for hidden in all_possible_hidden_states:
        transition_m[hidden] = transition_model(hidden)
        observation_m[hidden] = observation_model(hidden)
    
    for time in range(num_time_steps):
        w[time] = rover.Distribution()
        phi[time] = {}
        for z in all_possible_hidden_states:
            if observations[time] == None:
                prob = 1
            else:
                prob = observation_m[z][observations[time]]
            if time == 0 and prob != 0 and prior_distribution[z] != 0:
                w[time][z] = np.log(prob) + np.log(prior_distribution[z])
                continue
            elif time != 0:
                maximum = -np.inf
                for zn in w[time-1]:
                    if transition_m[zn][z] != 0:
                        new_maximum = np.log(transition_m[zn][z]) + w[time-1][zn]
                        if new_maximum > maximum and prob != 0:
                            maximum = new_maximum
                            phi[time][z] = zn
                if prob != 0:
                    w[time][z] = np.log(prob) + maximum
        
    w_max = -np.inf
    for z in w[num_time_steps-1]:
        new_max = w[num_time_steps-1][z]
        if new_max > w_max:
            w_max = new_max
            estimated_hidden_states[num_time_steps-1] = z
    for time in range(num_time_steps-2,-1,-1):
        estimated_hidden_states[time] = phi[time+1][estimated_hidden_states[time+1]]

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
    
    hidden_states = np.array(hidden_states)
    estimated_states = np.array(estimated_states)
    equal = (hidden_states == estimated_states).all(axis=1)
    print('viterbi error: ' + str(1-equal.sum()/num_time_steps))

    predictions = [max(marginals[i], key=marginals[i].get) for i in range(len(marginals))]
    equal = (hidden_states == predictions).all(axis=1)
    print('forward and backward error: ' + str(1-equal.sum()/num_time_steps))

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
        
