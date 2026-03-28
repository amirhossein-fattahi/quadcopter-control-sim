from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Common controller interface.

    All controllers should return:
        control: np.ndarray shape (4,)
        debug: dict
    """

    def __init__(self, params):
        self.params = params

    def reset(self):
        """
        Optional controller state reset.
        """
        pass

    @abstractmethod
    def compute(self, state, ref, dt=None):
        """
        Args:
            state: current quadcopter state
            ref: reference dictionary
            dt: optional timestep (needed for integral or predictive controllers)

        Returns:
            control, debug
        """
        raise NotImplementedError