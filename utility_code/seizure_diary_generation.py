import numpy as np


def generate_seizure_diary(num_months, 
                           monthly_mean, 
                           monthly_std_dev, 
                           time_scaling_const):
    '''

    This function generates one numpy array representing a seizure diary. Assuming the
    time_scaling_const parameter is set to 1, then each number in the array is a monthly seizure 
    count. The monthy seizure counts can be specified to have a specific mean and standard deviation,
    referred to here as the monthy mean and monthly standard deviation. The seizure counts are 
    generated according to a negative binomial distribution, implemented here via a gamma-poisson
    mixture.

    If a smaller time scale is needed (e.g., either weekly or daily seizure counts are needed instead),
    then the time_scaling_const parameter can be set to a number equal to the number of smaller time 
    units in a month. The array returned by this function will then be larger by a factor of 
    time_scaling_const. Also, the monthly mean will get scaled down by a factor of 1/time_scaling_const, 
    and the monthly standard deviation will also get scaled down by a factor of the square root of
    1/time_scaling_const.

    For example, requesting a seizure diary with 3 months and a time scaling constant of 1 will return a
    an array with 3 numbers representing monthly seizure counts. To get weekly seizure counts instead, the
    time_scaling_const parameter can be set to 4 (since there are 4 weeks in a month), and then an array of
    12 numbers representing 3 months' worth of weekly seizure counts will be returned instead. For daily 
    seizure counts, the time_scaling_const parameter can be set to 28 (28 days in a month), and then an array
    of 84 numbers (3 months' worth of daily seizure counts) will be returned.

    Inputs:
        1) num_months:
            (int) - the number of months in the seizure diary to be generated
        2) monthly_mean:
            (float) - the true monthly mean for the seizure diary to be generated
        3) monthly_std_dev:
            (float) - the true monthly standard deviation for the seizure diary to be generated
        4) time_scaling_const:
            (int) - the time-scaling factor which determines whether or not the seizure diary will either be 
                    generated on a monthly time scale, or if will be generated on a smaller time scale instead

                                monthly time scale --> testing_time_scaling_const = 1
                                weekly time scale  --> testing_time_scaling_const = 4  (4 weeks in a month)
                                daily time scale   --> testing_time_scaling_const = 28 (28 days in a month)

    Outputs:
        1) seizure_diary:
            (1D Numpy array) - an array of numbers representing a seziure diary

    '''

    # figure out the total number of seizure counts needed according to both the specified time scale and the specified number of months
    num_scaled_time_units = num_months*time_scaling_const

    # initialize the seizure diary array
    seizure_diary = np.zeros(num_scaled_time_units)

    # convert the monthly mean and monthly standard deviation into quantities usable by a gamma-poisson mixture
    monthly_var = np.power(monthly_std_dev, 2)
    monthly_mean_sq  = np.power(monthly_mean, 2)
    monthly_overdispersion = (monthly_var - monthly_mean)/monthly_mean_sq
    monthly_n = 1/monthly_overdispersion
    odds_ratio = monthly_overdispersion*monthly_mean

    # for each scaled time unit
    for scaled_time_unit_index in range(num_scaled_time_units):

        # generate seizure counts
        time_scaled_rate = np.random.gamma(monthly_n/time_scaling_const, odds_ratio)
        time_scaled_count = np.random.poisson(time_scaled_rate)

        # store the seizure counts
        seizure_diary[scaled_time_unit_index] = time_scaled_count
    
    return seizure_diary


def apply_effect(seizure_diary,
                 num_months,
                 time_scaling_const,
                 effect):
    '''

    This function applies a probabilistic drug effect to a given seizure diary which either
    reduces or increases the seizure counts, depedning on whether or the drug effect is 
    postivie or negative, respectively.

    Inputs:

        1) seizure_diary:
            (1D Numpy array) - an array of integers representing one seizure diary
        2) num_months:
            (int) - the number of months' worth of data in the seizure diary
        3) time_scaling_const:
            (int) - the time scale of the seizure diary
        4) effect:
            (float) - the percent size by which the seizure counts will be reduced

    Outputs:

        1) seizure_diary:
            (1D Numpy array) - the original seizure diary, except with probabilistically 
                               reduced seizure counts

    '''

    # figure out the number of scaled time units in the seizure diary
    num_scaled_time_units = num_months*time_scaling_const

    # for each scaled time unit
    for scaled_time_unit_index in range(num_scaled_time_units):

        # initialize the number of removed seizures for each seizure count
        num_removed = 0

        # for each seizure count
        for seizure_index in range(np.int_(seizure_diary[scaled_time_unit_index])):

            # generate a random floating-point number between 0 and 1
            prob = np.random.uniform(0, 1)

            # if the random number is lees than the effect,...
            if(prob < np.abs(effect)):
                
                # then say that a seizure has either been removed or added, depending on the postivity/negativity of the effect
                num_removed = num_removed + np.sign(effect)
        
        # actually remove (or add) the number of seizures which was probabilistically determined for this seizure count 
        seizure_diary[scaled_time_unit_index] = seizure_diary[scaled_time_unit_index] - num_removed

    return seizure_diary


def generate_seizure_diary_with_minimum_count(num_months,
                                              monthly_mean,
                                              monthly_std_dev,
                                              time_scaling_const,
                                              minimum_required_seizure_count):
    '''

    This function generates a seizure diary with a minimum number of seizures
    distributed over its seizure counts. This is meant to provide seizure diaries
    that can satisfy specific eligibility criteria.

    Inputs:

        1) num_months:
            (int) - the number of months in the seizure diary to be generated
        2) monthly_mean:
            (float) - the true monthly mean for the seizure diary to be generated
        3) monthly_std_dev
            (float) - the true monthly standard deviation for the seizure diary to be generated
        4) time_scaling_const
            (int) - the time-scaling factor which determines whether or not the seizure diary will either be 
                    generated on a monthly time scale, or if will be generated on a smaller time scale instead.
                    Affects the length of the seizure diary (see generate_seizure_diary for more details)

                                monthly time scale --> testing_time_scaling_const = 1
                                weekly time scale  --> testing_time_scaling_const = 4  (4 weeks in a month)
                                daily time scale   --> testing_time_scaling_const = 28 (28 days in a month)

        5) minimum_required_seizure_count:
            (int) - the minimum number of seizures that the diary will be generated with: the seizures
                    will be distributed over the seizure counts
    
    Outputs:

        1) seizure_diary_with_min_count:
            (1D Numpy array) - an array of numbers representing a seizure diary; the seizure diary is 
                               guaranteed to have a minimum number of seizures distributed over its
                               seizure counts

    '''

    # initialize the boolean condition for the upcoming while loop
    acceptable_count = False

    # while a seizure diary with an acceptable number of seizures has not been generated
    while(not acceptable_count):

        # generate a seizure diary
        seizure_diary_with_min_count = \
            generate_seizure_diary(num_months, 
                                   monthly_mean, 
                                   monthly_std_dev, 
                                   time_scaling_const)
    
        # count the number of seizures over that seizure diary
        num_seizures = np.sum(seizure_diary_with_min_count)

        # if it contains a number of seizures which is equal or greater than the minimum specified by the parameters,...
        if(num_seizures >= minimum_required_seizure_count):

            #.., then change the boolean condition to end the while loop
            acceptable_count = True
    
    return seizure_diary_with_min_count


def generate_baseline_seizure_diary(monthly_mean, 
                                    monthly_std_dev,
                                    num_baseline_months,
                                    baseline_time_scaling_const,
                                    minimum_required_baseline_seizure_count):
    '''

    This function is a wrapper function which generates the portion of a seizure diary that 
    corresponds to the baseline period. The seizure diary can be generated such that it 
    fulfills the eligibility criteria of having a minimum seizure count in the baseline period.

    There are two main advantages to generating the baseline and testing periods of a seizure diary
    seperately: firstly, calculation of endpoints are typically calculated separately over the 
    baseline and testing periods of a patient's seizure diary, so generating them separately and
    doing the required calculations over both periods is simpler than keeping trying to keep track 
    of where the baseline and testing seizure counts are in an array.

    Secondly, algorithms which only apply to one period (i.e., minimum required seizure count in baseline,
     drug effect in testing) only have to be applied to one array instead of applied to half of an
    array, thus reducing the complexity of the code.

    Thirdly, for the TTP endpoint, it's more computationally efficient since baseline seizure counts
    can be on a monthly time scale instead of a daily time scale.

    Inputs:

        1) monthly_mean:
            (float) - the true monthly mean for the baseline seizure diary to be generated
        2) monthly_std_dev:
            (float) - the true monthly standard deviation for the seizure diary to be generated
        3) num_baseline_months:
            (int) - the number of months in the baseline seizure diary to be generated
        4) baseline_time_scaling_const:
            (int) - the time-scaling factor which determines whether or not the baseline seizure diary 
                    will either be generated on a monthly time scale, or if will be generated on a 
                    smaller time scale instead.

                                monthly time scale --> testing_time_scaling_const = 1
                                weekly time scale  --> testing_time_scaling_const = 4  (4 weeks in a month)
                                daily time scale   --> testing_time_scaling_const = 28 (28 days in a month)

        5) minimum_required_baseline_seizure_count:
            (int) - the minimum number of seizures that the diary will be generated with: the seizures
                    will be distributed over the seizure counts
    
    Outputs:

        1) baseline_seizure_diary:
            (1D Numpy array) - an array of numbers representing the baseline period of a seizure diary; 
                               the seizure diary is guaranteed to have a minimum number of seizures 
                               distributed over its seizure counts

    '''

    baseline_seizure_diary = \
        generate_seizure_diary_with_minimum_count(num_baseline_months, 
                                                  monthly_mean, 
                                                  monthly_std_dev, 
                                                  baseline_time_scaling_const,
                                                  minimum_required_baseline_seizure_count)
    
    return baseline_seizure_diary


def generate_placebo_arm_testing_seizure_diary(num_testing_months, 
                                               monthly_mean, 
                                               monthly_std_dev, 
                                               testing_time_scaling_const,
                                               placebo_mu, 
                                               placebo_sigma):
    '''

    This function generates the testing period of a seizure diary for a patient
    who's been randomized to the placebo arm of a clinical trial.

    Only the placebo effect is applied to a seizure diary in the placebo arm. In this
    case, the placebo effect is implemented via the apply_effect() function. The effect
    parameter given to the apply_effect() function is generated here via a gaussian 
    distribution for each individual patient's seizure diary testing period.

    Inputs:

        1) num_testing_months:
            (int) - the number of months in the testing seizure diary to be generated
        2) monthly_mean:
            (float) - the true monthly mean for the testing seizure diary to be generated
        3) monthly_std_dev:
            (float) - the true monthly standard deviation for the testing seizure diary to be generated
        4) testing_time_scaling_const:
            (int) - the time-scaling factor which determines whether or not the testing seizure diary 
                    will either be generated on a monthly time scale, or if will be generated on a 
                    smaller time scale instead.

                                monthly time scale --> testing_time_scaling_const = 1
                                weekly time scale  --> testing_time_scaling_const = 4  (4 weeks in a month)
                                daily time scale   --> testing_time_scaling_const = 28 (28 days in a month)

        5) placebo_mu:
            (float) - the mean of the normally distributed placebo effect, expressed as a percentage
        6) placebo_sigma:
            (float) - the standard deviation of the normally distributed placebo effect, expressed as 
                      a percentage
    
    Outputs:

        1) placebo_arm_testing_seizure_diary:
            (1D Numpy array) - an array consisting of seizure counts representing the testing period of a 
                               seizure diary randomized to the placebo arm

    '''

    # generate the testing period of a seizure diary which was randomized to the placebo arm
    placebo_arm_testing_seizure_diary = \
        generate_seizure_diary(num_testing_months, 
                               monthly_mean, 
                               monthly_std_dev, 
                               testing_time_scaling_const)
    
    # generate this individual seizure diary's placebo effect according to the normal distribution
    placebo_effect = np.random.normal(placebo_mu, placebo_sigma)

    # apply the placebo effect to the testing period
    placebo_arm_testing_seizure_diary = \
        apply_effect(placebo_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     placebo_effect)

    return placebo_arm_testing_seizure_diary


def generate_drug_arm_testing_seizure_diary(num_testing_months, 
                                            monthly_mean, 
                                            monthly_std_dev, 
                                            testing_time_scaling_const,
                                            placebo_mu, 
                                            placebo_sigma,
                                            drug_mu, 
                                            drug_sigma):
    '''

    This function generates the testing period of a seizure diary for a patient
    who's been randomized to the drug arm of a clinical trial.

    The placebo effect is applied first to a seizure diary. After that, the drug effect
    is then sequentially applied as well. All effects are generated via gaussian distributions
    whose parameters are passed on to this function as inputs. Botht the drug effect and the
    placebo effect are implemented via the apply_effect() function.

    Inputs:

        1) num_testing_months:
            (int) - the number of months in the testing seizure diary to be generated
        2) monthly_mean:
            (float) - the true monthly mean for the testing seizure diary to be generated
        3) monthly_std_dev:
            (float) - the true monthly standard deviation for the testing seizure diary to be generated
        4) testing_time_scaling_const:
            (int) - the time-scaling factor which determines whether or not the testing seizure diary 
                    will either be generated on a monthly time scale, or if will be generated on a 
                    smaller time scale instead.

                                monthly time scale --> testing_time_scaling_const = 1
                                weekly time scale  --> testing_time_scaling_const = 4  (4 weeks in a month)
                                daily time scale   --> testing_time_scaling_const = 28 (28 days in a month)

        5) placebo_mu:
            (float) - the mean of the normally distributed placebo effect, expressed as a percentage
        6) placebo_sigma:
            (float) - the standard deviation of the normally distributed placebo effect, expressed as 
                      a percentage
        7) drug_mu:
            (float) - the mean of the normally distributed drug effect, expressed as a percentage
        8) drug_sigma:
            (float) - the standard deviation of the normally distributed drug effect, expressed as 
                      a percentage
    
    Outputs:

        1) drug_arm_testing_seizure_diary
            (1D Numpy array) - an array consisting of seizure counts representing the testing period of a 
                               seizure diary randomized to the drug arm

    '''

    # generate the testing period of a seizure diary which was randomized to the drug arm
    drug_arm_testing_seizure_diary = \
        generate_seizure_diary(num_testing_months, 
                               monthly_mean, 
                               monthly_std_dev, 
                               testing_time_scaling_const)

    # generate this individual seizure diary's placebo effect and drug effect, both generated according to the normal distribution
    placebo_effect = np.random.normal(placebo_mu, placebo_sigma)
    drug_effect    = np.random.normal(drug_mu,    drug_sigma)

    # apply the placebo effect to the testing period
    drug_arm_testing_seizure_diary = \
        apply_effect(drug_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     placebo_effect)
    
    # apply the drug effect to the testing period
    drug_arm_testing_seizure_diary = \
        apply_effect(drug_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     drug_effect)
    
    return drug_arm_testing_seizure_diary

