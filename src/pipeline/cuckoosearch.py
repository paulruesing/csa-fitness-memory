import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from math import gamma

from src.utils import plotting


class FitnessRAM:
    """
    A class to manage a fixed-size memory for storing fitness values associated with nests.

    Attributes
    ----------
    size : int
        The maximum number of entries (nests) that can be stored in the fitness memory.
    fitness_frame : pd.DataFrame
        A DataFrame to store nests and their corresponding fitness values.
    __index_next_overwrite : int
        An auxiliary index used to manage the FIFO (First-In-First-Out) memory when it is full.

    Parameters
    ----------
    size : int
        The maximum size of the fitness memory.
    """
    def __init__(self, size):
        """
        Initialize the FitnessRAM with the specified size.

        Parameters
        ----------
        size : int
            The maximum number of entries (nests) that can be stored.
        """
        self.size = size
        self.fitness_frame = pd.DataFrame(columns=['Nest', 'Fitness'],
                                          index=range(size))
        self.__index_next_overwrite = 0  # auxiliary for FIFO memory

    def write(self, nest, fitness_value):
        """
        Write the fitness value for a specified nest to the memory.

        If the memory is not full, the fitness value is added to the next available slot.
        If the memory is full, it replaces the oldest entry based on FIFO principle.

        Parameters
        ----------
        nest : any
            The identifier for the nest whose fitness value is to be recorded.
        fitness_value : float
            The fitness value to be stored for the specified nest.
        """
        nan_count = self.fitness_frame.Nest.isnull().sum()
        if nan_count != 0:  # until memory is filled, fill NaN elements from top to bottom
            self.fitness_frame.iloc[-nan_count, :] = nest, fitness_value
        else:  # FIFO principle for delete and refill if memory is full:
            self.fitness_frame.iloc[self.__index_next_overwrite, :] = nest, fitness_value
            # increase FIFO index by 1 but make it stay within length bound:
            self.__index_next_overwrite = (self.__index_next_overwrite + 1) % self.size

    def read(self, nest):
        """
        Retrieve the fitness value associated with a specified nest.

        Parameters
        ----------
        nest : any
            The identifier for the nest whose fitness value is to be retrieved.

        Returns
        -------
        float or None
            The fitness value for the specified nest if it exists; None if the nest is not found.
        """
        nest_index = None
        for ind, row in self.fitness_frame.iterrows():
            if np.isnan(row.iloc[0]).all():  # skip empty memory slots
                continue
            try:
                if row.iloc[0].equals(nest):  # save position if found
                    nest_index = ind
            except AttributeError:  # skip elements that are no series
                continue
        if nest_index is None:
            return None
        else:
            return self.fitness_frame.iloc[nest_index, 1]


class CuckooSearch:
    """
    Implements the **Cuckoo Search algorithm for optimization**.

    This class provides methods for initializing nests, performing fitness evaluations,
    and iterating the Cuckoo Search process to find optimal solutions based on provided fitness functions.
    The **parameters are derived from the series' indices that define the upper and lower bounds**.

    Attributes
    ----------
    no_of_nests : int
        The number of nests used in the search algorithm.
    lower_bounds : pd.Series
        The lower bounds for the parameters in the optimization problem. The series' index defines the parameter names.
    upper_bounds : pd.Series
        The upper bounds for the parameters in the optimization problem. The series' index defines the parameter names.
    fitness_function_list : list
        A list of fitness functions used to evaluate the nests.
    fitness_weight_list : list
        A list of weights corresponding to the fitness functions (default is equal weights).
    beta : float
        Parameter controlling the scale of the random walk in the algorithm.
    pa : float
        Probability of abandoning a nest and creating a new one.
    max_iterations : int
        The maximum number of iterations for the optimization process.
    verbose : bool
        If True, prints progress updates during execution.
    output_path : str or None
        Path to save results, if specified.
    use_fitness_memory : bool
        Indicates if fitness memory is being utilized.
    nest_frame : pd.DataFrame
        A DataFrame representing the current nests and their parameters.
    fitness_memory : pd.DataFrame
        The DataFrame containing fitness values for each nest.
    best_nest : pd.Series
        The parameters of the best nest found during the optimization.

    Parameters
    ----------
    lower_bounds : pd.Series
        The lower bounds for the parameters to be optimized. The series' index defines the parameter names.
    upper_bounds : pd.Series
        The upper bounds for the parameters to be optimized. The series' index defines the parameter names.
    fitness_function_list : list
        A list of fitness functions to evaluate the nests.
    fitness_weight_list : list, optional
        Weights for each fitness function (default is None, which assigns equal weights).
    no_of_nests : int, optional
        The number of nests to use (default is 10).
    beta : float, optional
        Parameter for the Levy flight distribution (default is 1.5).
    pa : float, optional
        Probability for abandoning a nest (default is 0.05).
    max_iterations : int, optional
        The maximum number of iterations to run the algorithm (default is 100).
    verbose : bool, optional
        If True, prints updates during execution (default is True).
    output_path : str, optional
        Directory path to save output files (default is None).
    fitness_memory_size : int, optional
        Size of the fitness memory to store fitness evaluations (default is None).
    direction : str, default is "minimzation"
            Defines the target of the optimzation (maximum or minimum fitness).

    Raises
    ------
    AssertionError
        If the lengths of lower and upper bounds do not match during initialization.
    """
    def __init__(self,
                 lower_bounds,
                 upper_bounds,
                 fitness_function_list,
                 fitness_weight_list=None,
                 no_of_nests=10,
                 beta=1.5,
                 pa=0.05,
                 max_iterations=100,
                 verbose=True,
                 output_path=None,
                 fitness_memory_size=None,
                 direction='minimization'
                 ) -> None:
        """
        Initialize the Cuckoo Search with specified parameters.

        Parameters
        ----------
        lower_bounds : pd.Series
            The lower bounds for the parameters to be optimized. The series' index defines the parameter names.
        upper_bounds : pd.Series
            The upper bounds for the parameters to be optimized. The series' index defines the parameter names.
        fitness_function_list : list
            A list of fitness functions to evaluate the nests.
        fitness_weight_list : list, optional
            Weights for each fitness function (default is None, which assigns equal weights).
        no_of_nests : int, optional
            The number of nests to use (default is 10).
        beta : float, optional
            Parameter for the Levy flight distribution (default is 1.5).
        pa : float, optional
            Probability for abandoning a nest (default is 0.05).
        max_iterations : int, optional
            The maximum number of iterations to run the algorithm (default is 100).
        verbose : bool, optional
            If True, prints updates during execution (default is True).
        output_path : str, optional
            Directory path to save output files (default is None).
        fitness_memory_size : int, optional
            Size of the fitness memory to store fitness evaluations (default is None).
        direction : str, default is "minimzation"
            Defines the target of the optimzation (maximum or minimum fitness).
        """
        self.no_of_nests = no_of_nests
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        assert len(lower_bounds) == len(upper_bounds), "Amount of lower and upper bounds needs to be equal!"
        self.no_of_parameters = len(lower_bounds)

        self.fitness_function_list = fitness_function_list
        if fitness_weight_list is None:
            self.fitness_weight_list = [1] * len(self.fitness_function_list)

        self.beta = beta
        self.pa = pa
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.output_path = output_path

        if fitness_memory_size is not None:
            self._fitness_memory = FitnessRAM(fitness_memory_size)
            self.use_fitness_memory = True
            print(f"Utilising fitness RAM with {fitness_memory_size} slots.")
        else:
            self.use_fitness_memory = False

        self.direction = direction
        self._best_nest = None  # working with private definition ("__") because @property decorator

        # nest frame with parameters as rows and nests as columns:
        # random number initialisation:
        self.nest_frame = pd.DataFrame(np.random.uniform(0, 1, (len(lower_bounds), no_of_nests)),
                                       index=lower_bounds.index,
                                       columns=pd.Series(range(no_of_nests), name="Nest"))

        # adjustment for bounds:
        for param_ind, (lower_bound, upper_bound) in enumerate(zip(lower_bounds, upper_bounds)):
            self.nest_frame.iloc[param_ind, :] = lower_bound + self.nest_frame.iloc[param_ind, :] * (
                        upper_bound - lower_bound)

    @property
    def fitness_memory(self):
        """
        Get the fitness memory DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing fitness values for each nest.
        """
        return self._fitness_memory.fitness_frame

    def fitness(self, nest):
        """
        Evaluate the fitness of a specified nest.

        If fitness memory is enabled, it first checks for the fitness value in memory.
        If not found, it computes the fitness using the provided fitness functions and updates the memory.

        Parameters
        ----------
        nest : pd.Series
            The parameters of the nest to be evaluated.

        Returns
        -------
        float
            The computed fitness value for the specified nest.
        """
        # read fitness memory:
        if self.use_fitness_memory:
            memory_output = self._fitness_memory.read(nest)
            if memory_output is not None:
                return memory_output

        fitness_value = 0
        for i, function in enumerate(self.fitness_function_list):
            fitness_value += function(nest) * self.fitness_weight_list[i]

        # write fitness memory:
        if self.use_fitness_memory:
            self._fitness_memory.write(nest, fitness_value)

        return fitness_value

    def return_optimal(self, nest1, nest2):
        ''' Compare two nests and return the better one based on fitness and the defined direction. Default is minimization. '''
        if self.direction == 'minimization':
            if self.fitness(nest1) < self.fitness(nest2):
                return nest1
            else:
                return nest2
        elif self.direction == 'maximization':
            if self.fitness(nest1) > self.fitness(nest2):
                return nest1
            else:
                return nest2
        else:
            raise ValueError(f"Direction {self.direction} not recognized. Has to be 'minimization' or 'maximization'!")

    def levy_flight(self):
        """
        Perform a Lévy flight to update the nests based on a random walk.

        This method applies a random step influenced by the current best solution to explore
        new solutions in the parameter space.

        Notes
        -----
        - The step size is influenced by the distance to the best known solution.
        - Utilizes statistical formulas from Levy flight distribution.

        Formulas (from DOI: 10.1109/NABIC.2009.5393690)
        --------
            X = X * N(0,1) * C      WHERE X IS THE NEST

            C = 0.01*S*(X-best)     WHERE S IS THE RANDOM STEP, and β = beta

                  u
            S = -----
                    1/β
                 |v|

            u ~ N(0,σu)     # NORMAL DISTRIBUTION WITH ZERO MEAN AND 'σu' STANDARD DEVIATION

                               Γ(1+β)*sin(πβ/2)
            with  σu^β =  --------------------------
                          Γ((1+β)/2)*β*(2^((β-1)/2))

            v ~ N(0,σv)

            with  σv = 1
        """
        # calculate sigma_u^β
        numerator = gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        denominator = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        # calculate sigma_u
        sigma_u = (numerator / denominator) ** (1 / self.beta)

        # sigma_v defined as 1
        sigma_v = 1

        # draws from normal distribution
        u = np.random.normal(0, sigma_u, self.no_of_parameters)
        v = np.random.normal(0, sigma_v, self.no_of_parameters)

        # random step S:
        S = u / (np.abs(v) ** (1 / self.beta))

        # update current best solution:
        self.identify_best()

        # create new instance of frame
        temp_nest_frame = self.nest_frame.copy()

        # apply random step and keep new result if improving:
        for i in range(self.no_of_nests):
            # combining everything as defined above
            # incorporating distance from best known (yielding no change for best)
            temp_nest_frame.iloc[:, i] += np.random.randn(self.no_of_parameters) * 0.01 * S * (
                        temp_nest_frame.iloc[:, i] - self.best_nest)

            # add only improving nests:
            self.nest_frame.iloc[:, i] = self.return_optimal(self.nest_frame.iloc[:, i], temp_nest_frame.iloc[:, i])

    def abandon_worst_nests(self):
        """
        Abandon the worst nests based on their fitness and replace them with new candidate nests.

        This method generates new nests by combining parameters from randomly selected nests.
        The new nests are evaluated, and only improving nests replace the old ones.
        """
        # save current configuration
        old_frame = self.nest_frame.copy()
        new_frame = self.nest_frame.copy()

        for nest_ind in range(self.no_of_nests):  # iterate over nests
            # randomly define two nests of which to derive spread for new nest value
            parent_nest_1, parent_nest_2 = np.random.randint(0, self.no_of_nests - 1, 2)
            for parameter_ind in range(self.no_of_parameters):
                r = np.random.rand()
                if r < self.pa:
                    # with pa probability (if r<pa) adjust new frame:
                    new_frame.iloc[parameter_ind, nest_ind] += np.random.rand() * (
                                old_frame.iloc[parameter_ind, parent_nest_1] - old_frame.iloc[
                            parameter_ind, parent_nest_2])

            # add only improving nests:
            self.nest_frame.iloc[:, nest_ind] = self.return_optimal(self.nest_frame.iloc[:, nest_ind],
                                                                    new_frame.iloc[:, nest_ind])

    @property
    def best_nest(self):
        """
        Get the current best nest found in the optimization process.
        If the best nest has not been identified, this method will call `identify_best()` to find it.
        """
        if self._best_nest is None:
            self.identify_best()
        return self._best_nest

    def identify_best(self):
        ''' Identifies current best solution based on fitness. '''
        for i in range(self.no_of_nests):
            # iterate through nests and keep best known by element-wise comparison
            if i == 0:
                self._best_nest = self.nest_frame.iloc[:, i]
            else:
                self._best_nest = self.return_optimal(self.best_nest, self.nest_frame.iloc[:, i])

    def enforce_bounds(self):
        """
        Ensure that the parameters of all nests stay within the defined bounds.
        This method clips the values of each parameter in the nests to be within the specified lower
        and upper bounds.
        """
        for param_ind, (lower_bound, upper_bound) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            self.nest_frame.iloc[param_ind, :] = np.clip(self.nest_frame.iloc[param_ind, :], lower_bound, upper_bound)

    def execute(self, iterations=None, plot=True, save_step=None):
        """
        Execute the Cuckoo Search algorithm.

        This method performs the main optimization process, updating nests, evaluating fitness,
        and optionally saving results and plotting progress.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations to run the algorithm (default is self.max_iterations).
        plot : bool, optional
            If True, plots the progress of the search (default is True).
        save_step : int or None, optional
            The frequency of saving results during execution (default is None).
        """
        self.time_list, self.fitness_list = list(), list()

        if iterations is None:
            iterations = self.max_iterations

        # detect wrong specifications
        if (save_step is not None) & (self.output_path is None):
            print(f"Problem: save_step was defined but no output_path provided. Saving not possible!")
            save_step = None

        for step in tqdm(range(1, iterations + 1)):
            # random permutations:
            self.levy_flight()
            self.enforce_bounds()
            self.abandon_worst_nests()
            self.enforce_bounds()

            # save progress:
            self.time_list.append(step)
            best_fitness = self.fitness(self.best_nest)
            self.fitness_list.append(best_fitness)

            # status message
            if self.verbose:
                print(f"Iteration:            {step} | Best current fitness:     {best_fitness}")

            # save current best result:
            if save_step is not None:
                if (step + 1) % save_step == 0:
                    self.best_nest.to_csv(self.output_path /
                                          plotting.file_title(title=f"Best nest (fit {round(best_fitness, 6)})",
                                                              dtype_suffix=".csv"))

        print(f'\nOptimal solution (with fitness: {best_fitness}):\n {self.best_nest}')

        if plot:
            self.plot_progress()

    def __call__(self, **kwargs):
        """
        Execute the Cuckoo Search algorithm using the provided keyword arguments.
        This method allows the class instance to be called as a function.
        """
        self.execute(**kwargs)

    def plot_progress(self):
        """
        Plot the progress of the Cuckoo Search algorithm based on fitness over iterations.

        This method displays a line plot of fitness values across iterations to visualize optimization progress.

        Raises
        ------
        BaseException
            If the search has not been carried out yet (i.e., no fitness values available).
        """
        # assert that search has been carried out:
        try:
            if self.fitness_list is None:
                raise BaseException("Search needs to be carried out through execute() first!")
        except AttributeError:
            raise BaseException("Search needs to be carried out through execute() first!")
        # create plot:
        fig, ax1 = plt.subplots(1, 1)
        sns.lineplot(x=self.time_list, y=self.fitness_list, ax=ax1)
        ax1.set_title("Cuckoo Search Results")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Fitness")