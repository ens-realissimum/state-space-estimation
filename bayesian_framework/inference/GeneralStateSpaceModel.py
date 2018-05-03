from abc import ABC, abstractmethod

from ..inference.validation_exceptions import CorruptedStateSpaceModelStructureException


class GeneralStateSpaceModel(ABC):
    """
    Model which allows to formulate filtration issue in bayesian estimation terminology.
    """

    @abstractmethod
    @property
    def type(self):
        return 'General state space model'

    @abstractmethod
    @property
    def tag(self):
        return ''

    @abstractmethod
    @property
    def state_dimension(self):
        msg = 'State dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    @property
    def observation_dimension(self):
        msg = 'Observation dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    @property
    def control_state_dimension(self):
        return None

    @abstractmethod
    @property
    def control_observation_dimension(self):
        return None

    @abstractmethod
    @property
    def state_noise_dimension(self):
        msg = 'State noise dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    @property
    def observation_noise_dimension(self):
        msg = 'Observation noise dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    @property
    def state_noise(self):
        msg = 'State noise dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    @property
    def observation_noise(self):
        msg = 'Observation noise dimension should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def transition_func(self):
        """
        State transition function. Calculate (predict) value of state space vector at time = k based on value of
        state space vector at time = k-1.
        :return: value of state space vector
        """
        msg = 'State transition function should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def observation_func(self):
        """
        Observation function (mapping state to observation)
        :return: observation calculated based on value of state space vector at time = k.
        """
        msg = 'Observation function (mapping state to observation) should be defined for state space model'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def prior(self):
        """
        Function to calculate that calculates P(x(k)|x(k-1)), ie probability that value of state space vector
        at time = k equal to 'x(k)' if at previous time the value has been equal to x(k-1).
        :return: probability that value of state space vector at time = k equal to 'x(k)' if at previous time
        the value has been equal to x(k-1).
        """
        msg = 'Function to calculate the state transition function that calculates P(x(k)|x(k-1))'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def likelihood(self):
        """
        Function to calculate the observation likelihood function that calculates p(z(k)|x(k))
        :return: calculated likelihood.
        """
        msg = 'Function to calculate the observation likelihood function that calculates p(z(k)|x(k))'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def innovation(self, observation, observation_predicted):
        """
        Function to calculate the innovation model function that calculates the difference between the output
        of the observation function (observation_func) and the actual 'real-world' measurement/observation
        of that signal.
        :param observation: 'real-world' measurement/observation of that signal at time = k
        :param observation_predicted: the output of the observation function (observation_func) at time = k.
        :return: difference between the output of the observation function and the actual 'real-world' observation.
        """
        return observation - observation_predicted

    @abstractmethod
    def linearize(self):
        """
        The linearization function that calculates Jacobians e.t.c.
        :return: value of the function calculated in the linear approximation.
        """
        msg = 'You are using algorithm which are required linearization, hence linearize should be defined'
        raise CorruptedStateSpaceModelStructureException(msg)

    @abstractmethod
    def set_params(self):
        """
        Allow to update specific model parameters. Do nothing by default
        :return:
        """
        pass
