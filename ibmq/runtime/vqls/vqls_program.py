# Grab functions and modules from dependencies
import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
import mthree

# Grab functions and modules from Qiskit needed
from qiskit import QuantumCircuit, transpile
import qiskit.circuit.library.n_local as lib_local
from qiskit.algorithms.optimizers import SPSA

from typing import Optional, Union, List, Callable, Tuple
import numpy as np


from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter 


from qiskit.algorithms.variational_algorithm import VariationalAlgorithm

from qiskit.providers import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from qiskit.utils.validation import validate_min

from qiskit.algorithms.linear_solvers.observables.linear_system_observable import LinearSystemObservable

from qiskit.opflow import (Z, I, StateFn, OperatorBase, TensoredOp, ExpectationBase,
                           CircuitSampler, ListOp, ExpectationFactory)

from qiskit.opflow import (CircuitSampler, ExpectationBase, ExpectationFactory,
                           ListOp, OperatorBase, StateFn)

from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer
from qiskit.opflow.gradients import GradientBase

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.algorithms.variational_algorithm  import VariationalResult
from qiskit.algorithms.linear_solvers.linear_solver import LinearSolverResult

from types import SimpleNamespace
import numpy as np 
import scipy.linalg as spla
from qiskit.circuit import QuantumCircuit

from typing import Optional, List, Union
from qiskit import QuantumCircuit

# The entrypoint for our Runtime Program
def main(
    backend,
    user_messenger,
    matrix,
    rhs,
    ansatz,
    x0=None,
    optimizer="SPSA",
    optimizer_config={"maxiter": 100},
    shots=8192,
    use_measurement_mitigation=False,
):

    """
    The main sample VQE program.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the
                                        program user.
        matrix (np.ndarray): the matrix of the linear system.
        rhs (np.ndarray): the right habd side of the linear system.
        ansatz (QuantumCircuit): the quantum circuit of the ansatz.
        x0 (array_like): Optional, initial vector of parameters.
        optimizer (str): Optional, string specifying classical optimizer,
                         default='SPSA'.
        optimizer_config (dict): Optional, configuration parameters for the
                                 optimizer.
        shots (int): Optional, number of shots to take per circuit.
        use_measurement_mitigation (bool): Optional, use measurement mitigation,
                                           default=False.

    Returns:
        OptimizeResult: The result in SciPy optimization format.
    """

    vqls = VQLS(ansatz=ansatz,
                optimizer=optimizer,
                quantum_instance=backend)

    full_circs = vqls.construct_circuit(matrix, rhs, appply_explicit_measurement=True)
    meas_strings = ['1'] * len(full_circs)

    # Get the number of parameters in the ansatz circuit.
    num_params = vqls.ansatz.num_parameters

    # Use a given initial state, if any, or do random initial state.
    if x0:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != num_params:
            raise ValueError(
                "Number of params in x0 ({}) does not match number \
                              of ansatz parameters ({})".format(
                    x0.shape[0], num_params
                )
            )
    else:
        x0 = 2 * np.pi * np.random.rand(num_params)

    # Because we are in general targeting a real quantum system, our circuits must be transpiled
    # to match the system topology and, hopefully, optimize them.
    # Here we will set the transpiler to the most optimal settings where 'sabre' layout and
    # routing are used, along with full O3 optimization.

    # This works around a bug in Qiskit where Sabre routing fails for simulators (Issue #7098)
    trans_dict = {}
    if not backend.configuration().simulator:
        trans_dict = {"layout_method": "sabre", "routing_method": "sabre"}
    trans_circs = transpile(full_circs, backend, optimization_level=3, **trans_dict)

    # If using measurement mitigation we need to find out which physical qubits our transpiled
    # circuits actually measure, construct a mitigation object targeting our backend, and
    # finally calibrate our mitgation by running calibration circuits on the backend.
    if use_measurement_mitigation:
        maps = mthree.utils.final_measurement_mapping(trans_circs)
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(maps)

    # Here we define a callback function that will stream the optimizer parameter vector
    # back to the user after each iteration.  This uses the `user_messenger` object.
    # Here we convert to a list so that the return is user readable locally, but
    # this is not required.
    def callback(xk):
        user_messenger.publish(list(xk))

    # This is the primary VQE function executed by the optimizer. This function takes the
    # parameter vector as input and returns the energy evaluated using an ansatz circuit
    # bound with those parameters.
    def vqls_func(params):

        # Attach (bind) parameters in params vector to the transpiled circuits.
        bound_circs = [circ.bind_parameters(params) for circ in trans_circs]

        # Submit the job and get the resultant counts back
        counts = backend.run(bound_circs, shots=shots).result().get_counts()

        # If using measurement mitigation apply the correction and
        # compute expectation values from the resultant quasiprobabilities
        # using the measurement strings.
        if use_measurement_mitigation:
            quasi_collection = mit.apply_correction(counts, maps)
            expvals = quasi_collection.expval(meas_strings)
        # If not doing any mitigation just compute expectation values
        # from the raw counts using the measurement strings.
        # Since Qiskit does not have such functionality we use the convenence
        # function from the mthree mitigation module.
        else:
            print(meas_strings)
            print(counts)
            expvals = mthree.utils.expval(counts, meas_strings)

        # get the probas
        probas = [vqls.get_probability_from_expected_value(e) for e in expvals]

        # The final cost is obatined via the vqls process_probability_circuit_output
        # method
        cost = vqls.process_probability_circuit_output(probas)
        return cost

    # Here is where we actually perform the computation.  We begin by seeing what
    # optimization routine the user has requested, eg. SPSA verses SciPy ones,
    # and dispatch to the correct optimizer.  The selected optimizer starts at
    # x0 and calls 'vqe_func' everytime the optimizer needs to evaluate the cost
    # function.  The result is returned as a SciPy OptimizerResult object.
    # Additionally, after every iteration, we use the 'callback' function to
    # publish the interim results back to the user. This is important to do
    # so that if the Program terminates unexpectedly, the user can start where they
    # left off.

    # Since SPSA is not in SciPy need if statement
    if optimizer == "SPSA":
        res = fmin_spsa(vqls_func, x0, args=(), **optimizer_config, callback=callback)
    # All other SciPy optimizers here
    else:
        res = opt.minimize(
            vqls_func, x0, method=optimizer, options=optimizer_config, callback=callback
        )
    # Return result. OptimizeResult is a subclass of dict.
    return res


def fmin_spsa(
    func,
    x0,
    args=(),
    maxiter=100,
    a=1.0,
    alpha=0.602,
    c=1.0,
    gamma=0.101,
    callback=None,
):
    """
    Minimization of scalar function of one or more variables using simultaneous
    perturbation stochastic approximation (SPSA).

    Parameters:
        func (callable): The objective function to be minimized.

                          ``fun(x, *args) -> float``

                          where x is an 1-D array with shape (n,) and args is a
                          tuple of the fixed parameters needed to completely
                          specify the function.

        x0 (ndarray): Initial guess. Array of real elements of size (n,),
                      where ‘n’ is the number of independent variables.

        maxiter (int): Maximum number of iterations.  The number of function
                       evaluations is twice as many. Optional.

        a (float): SPSA gradient scaling parameter. Optional.

        alpha (float): SPSA gradient scaling exponent. Optional.

        c (float):  SPSA step size scaling parameter. Optional.

        gamma (float): SPSA step size scaling exponent. Optional.

        callback (callable): Function that accepts the current parameter vector
                             as input.

    Returns:
        OptimizeResult: Solution in SciPy Optimization format.

    Notes:
        See the `SPSA homepage <https://www.jhuapl.edu/SPSA/>`_ for usage and
        additional extentions to the basic version implimented here.
    """
    A = 0.01 * maxiter
    x0 = np.asarray(x0)
    x = x0

    for kk in range(maxiter):
        ak = a * (kk + 1.0 + A) ** -alpha
        ck = c * (kk + 1.0) ** -gamma
        # Bernoulli distribution for randoms
        deltak = 2 * np.random.randint(2, size=x.shape[0]) - 1
        grad = (func(x + ck * deltak, *args) - func(x - ck * deltak, *args)) / (
            2 * ck * deltak
        )
        x -= ak * grad

        if callback is not None:
            callback(x)

    return OptimizeResult(
        fun=func(x, *args),
        x=x,
        nit=maxiter,
        nfev=2 * maxiter,
        message="Optimization terminated successfully.",
        success=True,
    )



class VariationalLinearSolverResult(LinearSolverResult, VariationalResult):
    """A base class for linear systems results using variational methods

    The  linear systems variational algorithms return an object of the type ``VariationalLinearSystemsResult``
    with the information about the solution obtained.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def cost_function_evals(self) -> Optional[int]:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def state(self) -> Union[QuantumCircuit, np.ndarray]:
        """return either the circuit that prepares the solution or the solution as a vector"""
        return self._state

    @state.setter
    def state(self, state: Union[QuantumCircuit, np.ndarray]) -> None:
        """Set the solution state as either the circuit that prepares it or as a vector.

        Args:
            state: The new solution state.
        """
        self._state = state

class VariationalLinearSolver(ABC):
    """An abstract class for linear system solvers in Qiskit."""

    @abstractmethod
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                BaseOperator,
                List[LinearSystemObservable],
                List[BaseOperator],
            ]
        ] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> VariationalLinearSolverResult:
        """Solve the system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is ``None``.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Returns:
            The result of the linear system.
        """
        raise NotImplementedError


class UnitaryDecomposition:
    # https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    
    def __init__(self, mat, thr=1E-6, check=True, normalize_coefficients=True):
        
        self.mat = mat
        self.unit_coeffs = None
        self.unit_mats = None 
        self.thr = thr
        self.num_qubits = int(np.log2(mat.shape[0]))

        coeffs, mats = self.decompose(check=check, normalize_coefficients=normalize_coefficients)
        circuits = self.get_circuits(mats)

        self.data = [SimpleNamespace(mat=m, coeff=c, circuit=qc) for m,c,qc in zip(mats, coeffs, circuits)]
        self.len_data = len(self.data)

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.len_data:
            out = self.data[self.iiter]
            self.iiter += 1 
            return out
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_real(x: np.ndarray) -> np.ndarray:
        """ Get the real part of x"""
        return 0.5 * (x + np.conjugate(x))

    @staticmethod
    def get_imag(x: np.ndarray) -> np.ndarray:
        """ Get the imaginary part of x """
        return 0.5/(1j) * (x - np.conjugate(x))

    @staticmethod
    def aux(x: np.ndarray) -> np.ndarray:
        """ Auxiliary function

        Performs a matrix operation that we'll need later
        """
        I = np.eye(len(x))
        x2 = np.linalg.matrix_power(x,2)
        return 1j*spla.sqrtm(I - x2)

    def decompose(self, check: bool = False, normalize_coefficients=False):

        """ Unitary decomposition

        Decomposes the complex normalized matrix X into four unitary matrices
        """

        # Normalize
        norm = np.linalg.norm(self.mat)
        X_n = self.mat / norm

        # Apply the algorithm as described in
        
        B = self.get_real(X_n)
        C = self.get_imag(X_n)

        ## Get the matrices
        UB = B + self.aux(B)
        UC = C + self.aux(C)
        VB = B - self.aux(B)
        VC = C - self.aux(C)

        ## Get the coefficients
        cb = norm * 0.5
        cc = cb * 1j

        if np.allclose(self.mat, self.mat.T.conj()):
            ## Return
            unit_coeffs = [cb,cb]
            unit_mats = [UB, VB]
        else:
            ## Return
            unit_coeffs = [cb,cb,cc,cc]
            unit_mats = [UB, VB, UC, VC]         

        if check:
            mat_recomp = self.recompose(unit_coeffs, unit_mats)
            assert(np.allclose(self.mat,mat_recomp))

        if normalize_coefficients:
            unit_coeffs = self.normalize_coefficients(unit_coeffs)

        return unit_coeffs, unit_mats

    def normalize_coefficients(self, unit_coeffs):
        """Normalize the coefficients
        """

        sum_coeff = np.array(unit_coeffs).sum()
        return [u/sum_coeff for u in unit_coeffs]



    def recompose(self, unit_coeffs, unit_mats):
        """ Rebuilds the original matrix from the decomposed one """
        recomp = np.zeros_like(unit_mats[0])
        for c, m in zip(unit_coeffs, unit_mats):
            recomp += c * m
        return recomp

    def get_circuits(self, unit_mats):
        """Construct the circuits
        """
        circuits = []
        for m in unit_mats:
            qc = QuantumCircuit(self.num_qubits)
            qc.unitary(m, qc.qubits)
            circuits.append(qc)

        return circuits


class HadammardTest:

    def __init__(
        self,
        operators: Union[QuantumCircuit, List[QuantumCircuit]],
        use_barrier: Optional[bool] = False,
        apply_control_to_operator: Optional[bool] = True,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False
    ) -> List[QuantumCircuit]:

        if isinstance(operators, QuantumCircuit):
            operators = [operators]

        if apply_control_to_operator:
            self.num_qubits = operators[0].num_qubits + 1
        else:
            self.num_qubits = operators[0].num_qubits
        self.num_clbits = 1 


        self.circuits = self._build_circuit(operators, 
                                            use_barrier, 
                                            apply_control_to_operator,
                                            apply_initial_state,
                                            apply_measurement)

        self.ncircuits = len(self.circuits) 


    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.ncircuits:
            out = self.circuits[self.iiter]
            self.iiter += 1
            return out
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, index):
        return self.circuits[index]

    def _build_circuit(self, 
                      operators: List[QuantumCircuit], 
                      use_barrier: bool, 
                      apply_control_to_operator: bool,
                      apply_initial_state: Optional[QuantumCircuit] = None,
                      apply_measurement: Optional[bool] = False
                      ) -> List[QuantumCircuit]:
        """_summary_

        Args:
            operators (List[QuantumCircuit]): _description_
            use_barrier (bool): _description_
            apply_control_to_operator (bool): _description_

        Returns:
            List[QuantumCircuit]: _description_
        """

        circuits = []
        
        for imaginary in [False, True]:

            if apply_measurement:
                qc = QuantumCircuit(self.num_qubits,self.num_clbits)
            else:
                qc = QuantumCircuit(self.num_qubits)

            
            if apply_initial_state is not None:
                qc.compose(apply_initial_state, 
                           list(range(1,self.num_qubits)), 
                           inplace=True)

            if use_barrier:
                qc.barrier()

            # hadadmard gate on ctrl qbit
            qc.h(0)

            # Sdg on ctrl qbit
            if imaginary:
                qc.sdg(0)
            
            if use_barrier:
                qc.barrier()

            # matrix circuit
            for op in operators:
                if apply_control_to_operator:
                    qc.compose(op.control(1),
                                qubits=list(range(0,self.num_qubits)),
                                inplace=True)
                else:
                    qc.compose(op,
                                qubits=list(range(0,self.num_qubits)),
                                inplace=True)                    
            if use_barrier:
                qc.barrier()

            # hadamard on ctrl circuit
            qc.h(0)

            # measure
            if apply_measurement:
                qc.measure(0,0)

            circuits.append(qc)

        return circuits
        


class VQLS(VariationalAlgorithm, VariationalLinearSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    Examples:

        .. jupyter-execute:

            from qiskit.algorithms.linear_solvers.vqls import VQLS
            from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
            from qiskit.algorithms.optimizers import COBYLA
            from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
            from qiskit import Aer
            import numpy as np

            # define the matrix and the rhs           
            matrix = np.random.rand(4,4)
            matrix = (matrix + matrix.T)
            rhs = np.random.rand(4)

            # number of qubits needed
            num_qubits = int(log2(A.shape[0]))

            # get the classical solution
            classical_solution = NumPyLinearSolver().solve(matrix,rhs/np.linalg.norm(rhs))     

            # specify the backend
            backend = Aer.get_backend('aer_simulator_statevector')

            # specify the ansatz
            ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=3, insert_barriers=False)

            # declare the solver
            vqls  = VQLS(
                ansatz=ansatz,
                optimizer=COBYLA(maxiter=200, disp=True),
                quantum_instance=backend
            )

            # solve the system
            solution = vqls.solve(matrix,rhs)

    References: 

        [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, Patrick J. Coles
        Variational Quantum Linear Solver
        `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>` 
    """

    def __init__(
            self,
            ansatz: Optional[QuantumCircuit] = None,
            optimizer: Optional[Union[Optimizer, Minimizer]] = None,
            initial_point: Optional[np.ndarray] = None,
            gradient: Optional[Union[GradientBase, Callable]] = None,
            expectation: Optional[ExpectationBase] = None,
            include_custom: bool = False,
            max_evals_grouped: int = 1,
            callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
            quantum_instance: Optional[Union[Backend, QuantumInstance]] = None
        ) -> None:
        r"""
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Three parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the cost and the optimizer parameters for the ansatz
            quantum_instance: Quantum Instance or Backend
        """
        super().__init__()


        validate_min("max_evals_grouped", max_evals_grouped, 1)

        self._num_qubits = None

        self._max_evals_grouped = max_evals_grouped
        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._include_custom = include_custom

        self._ansatz = None
        self.ansatz = ansatz

        self._initial_point = None
        self.initial_point = initial_point

        self._optimizer = None
        self.optimizer = optimizer

        self._gradient = None
        self.gradient = gradient

        self._quantum_instance = None

        if quantum_instance is None:
            quantum_instance = Aer.get_backend('aer_simulator_statevector')
        self.quantum_instance = quantum_instance

        self._expectation = None
        self.expectation = expectation 

        self._callback = None
        self.callback = callback

        self._eval_count = 0

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def num_clbits(self) -> int:
        """return the numner of classical bits"""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits: int) -> None:
        """Set the number of classical bits"""
        self._num_clbits = num_clbits

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz
        self.num_qubits = ansatz.num_qubits + 1

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, Backend]) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, 
            statevector=is_statevector_backend(quantum_instance.backend),
            param_qobj=is_aer_provider(quantum_instance.backend)
        )

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def max_evals_grouped(self) -> int:
        """Returns max_evals_grouped"""
        return self._max_evals_grouped

    @max_evals_grouped.setter
    def max_evals_grouped(self, max_evals_grouped: int):
        """Sets max_evals_grouped"""
        self._max_evals_grouped = max_evals_grouped
        self.optimizer.set_max_evals_grouped(max_evals_grouped)

    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]):
        """Sets callback"""
        self._callback = callback


    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used. If None is passed, SLSQP is used by default.

        """
        if optimizer is None:
            optimizer = SLSQP()

        if isinstance(optimizer, Optimizer):
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        self._optimizer = optimizer


    def construct_circuit(
        self,
        matrix: Union[np.ndarray, UnitaryDecomposition],
        vector: Union[np.ndarray, QuantumCircuit],
        appply_explicit_measurement: Optional[bool] = False
        ) -> List[QuantumCircuit]:
        """Returns the a list of circuits required to compute the expectation value

        Args:
            matrix (Union[np.ndarray, UnitaryDecomposition]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of thge linear system
            appply_explicit_measurement (bool, Optional): add the measurement operation in the circuits

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            List[QuantumCircuit]: Quantum Circuits required to compute the cost function
        """

        # state preparation
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            self.vector_circuit = vector

        elif isinstance(vector, np.ndarray):
            nb = int(np.log2(len(vector)))
            self.vector_circuit = QuantumCircuit(nb)
            self.vector_circuit.prepare_state(vector/np.linalg.norm(vector))

        # matrix preparation
        if isinstance(matrix, UnitaryDecomposition):    
            self.matrix_circuits = matrix 

        elif isinstance(matrix, np.ndarray):

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2**self.vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(self.vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )

            nb = int(np.log2(matrix.shape[0]))
            self.matrix_circuits = UnitaryDecomposition(matrix, 
                                                        check=True, 
                                                        normalize_coefficients=True)
        circuits = []
        self.num_hdmr = 0

        # compute only the circuit for <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        for ii in range(len(self.matrix_circuits)):
            mi = self.matrix_circuits[ii]

            for jj in range(ii+1,len(self.matrix_circuits)):
                mj = self.matrix_circuits[jj]
                circuits += HadammardTest(operators=[mi.circuit.inverse(), mj.circuit],
                                          apply_initial_state=self._ansatz,
                                          apply_measurement=appply_explicit_measurement)

                self.num_hdmr += 1
                
        for mi in self.matrix_circuits:
            circuits += HadammardTest(operators=[self.ansatz, mi.circuit, self.vector_circuit.inverse()],
                                      apply_measurement=appply_explicit_measurement) 


        return circuits   

    def construct_observalbe(self):
        """Create the operator to measure the circuit output.
        """

        # create thew obsevable 
        one_op = (I - Z) / 2
        self.observable = TensoredOp((self.num_qubits-1) * [I]) ^ one_op

    def construct_expectation(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        circuit: QuantumCircuit,
    ) -> Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.
            circuit: one of the circuit required for the cost calculation

        Returns:
            The Operator equalling the measurement of the circuit :class:`StateFn` by the
            observable's expectation :class:`StateFn`

        """

        # assign param to circuit
        wave_function = circuit.assign_parameters(parameter)

        # compose the statefn of the observable on the circuit
        return ~StateFn(self.observable) @ StateFn(wave_function)


    @staticmethod
    def get_probability_from_expected_value(exp_val: complex) -> float:
        """transform the state array of the circuit to a proba

        Args:
            exp_val (complex): expected value of the observable
        """
        return 1.0 - 2.0 * exp_val


    def get_hadamard_sum_coeffcients(self):
        """Compute the c_i^*c_i and  c_i^*c_j coefficients.
        """

         # compute all the ci.conj * cj  for i<j
        cii_coeffs, cij_coeffs = [], []
        for ii in range(len(self.matrix_circuits)):
            ci = self.matrix_circuits[ii].coeff
            cii_coeffs.append(ci.conj()*ci)
            for jj in range(ii+1,len(self.matrix_circuits)):
                cj = self.matrix_circuits[jj].coeff
                cij_coeffs.append(ci.conj()*cj)

        return np.array(cii_coeffs), np.array(cij_coeffs) 

    def process_probability_circuit_output(self, probabiliy_circuit_output: List) -> float:
        """Compute the final cost function from the output of the different circuits

        Args:
            probabiliy_circuit_output (List): expected values of the different circuits

        Returns:
            float: value of the cost function
        """

        # compute all the ci.conj * cj  for i<j
        cii_coeffs, cij_coeffs = self.get_hadamard_sum_coeffcients()

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        phi_phi_overlap  = self._compute_phi_overlap(cii_coeffs, cij_coeffs, probabiliy_circuit_output)

        # compute all the terms in |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
        b_phi_overlap = self._compute_bphi_overlap(cii_coeffs, cij_coeffs, probabiliy_circuit_output)

        # overall cost
        cost = 1.0 - np.real(b_phi_overlap/phi_phi_overlap)
        print('Cost function %f' %cost)
        return cost

    def _compute_phi_overlap(self, cii_coeff: np.ndarray, 
                            cij_coeff: np.ndarray, 
                            probabiliy_circuit_output: List) -> float:
        """Compute the state overlap

        .. math:
            \langle\Phi|\Phi\rangle = \sum_{nm} c_n^*c_m \langle 0|V^* U_n^* U_m V|0\rangle 

        Args:
            sum_coeff (List): the values of the c_n^* c_m coefficients
            probabiliy_circuit_output (List): the values of the \langle 0|V^* U_n^* U_m V|0\rangle terms

        Returns:
            float: value of the sum
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        phi_phi_overlap = np.array(probabiliy_circuit_output)[:2*self.num_hdmr]
        phi_phi_overlap = phi_phi_overlap.astype('complex128')
        phi_phi_overlap *= np.array([1.,1.j]*self.num_hdmr)
        phi_phi_overlap = phi_phi_overlap.reshape(-1,2).sum(1)

        phi_phi_overlap *= cij_coeff
        phi_phi_overlap = phi_phi_overlap.sum()
        phi_phi_overlap += phi_phi_overlap.conj()

        # add the diagonal terms
        # since <0|V Ai* Aj V|0> = 1 we simply
        # add the sum of the cici coeffs
        phi_phi_overlap += cii_coeff.sum()

        return phi_phi_overlap

    def _compute_bphi_overlap(self, cii_coeffs: np.ndarray, cij_coeffs: np.ndarray, 
                              probabiliy_circuit_output: List)-> float:
        """Compute the overlap 

        .. math:
            |\langle\b|\Phi\rangle|^2 = \sum_{nm} c_n^*c_m \langle 0|V^* U_n^* U_b |0\rangle \langle 0|U_b^* U_m V |0\rangle 
        Args:
            cii_coeffs (List): the values of the c_i^* c_i coeffcients
            cij_coeffs (List): the values of the c_i^* c_j coeffcients
            probabiliy_circuit_output (List): values of the \langle 0|V^* U_n^* U_b |0\rangle terms

        Returns:
            float: value of the sum
        """

        # compute <0|V* Ai* U|0> = p[k] + 1.0j p[k+1]
        # with k = 2*(self.num_hdmr + i)
        bphi_terms = np.array(probabiliy_circuit_output)[2*self.num_hdmr:]
        bphi_terms = bphi_terms.astype('complex128')
        bphi_terms *= np.array([1.,1.j] * int(len(bphi_terms)/2))
        bphi_terms = bphi_terms.reshape(-1,2).sum(1)

        # init the final result
        b_phi_overlap = 0.0 + 0.0j
        nterm = len(bphi_terms)
        iterm = 0
        for i in range(nterm):
            # add |c_i|^2 <0|V* Ai* U|0> * <0|U* Ai V|0>
            xii = cii_coeffs[i] * bphi_terms[i] * bphi_terms[i].conj()
            b_phi_overlap += xii
            for j in range(i+1, nterm):
                # add c_i* c_j <0|V* Ai* U|0> * <0|U* Aj V|0>
                xij = cij_coeffs[iterm] * bphi_terms[i] * bphi_terms[j].conj()
                b_phi_overlap += xij
                # add c_j* c_i <0|V* Aj* U|0> * <0|U* Ai V|0>
                b_phi_overlap += xij.conj()
                iterm += 1

        return b_phi_overlap


    def get_cost_evaluation_function(
        self,
        circuits: List[QuantumCircuit],
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Generate the cost function of the minimazation process

        Args:
            circuits (List[QuantumCircuit]): circuits necessary to compute the cost function

        Raises:
            RuntimeError: If the ansatz is not parametrizable

        Returns:
            Callable[[np.ndarray], Union[float, List[float]]]: the cost function
        """


        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        # construct the observable op
        self.construct_observalbe() 

        ansatz_params = self.ansatz.parameters
        expect_ops = []
        for circ in circuits:
            expect_ops.append(self.construct_expectation(ansatz_params, circ))

        # create a ListOp for performance purposes
        expect_ops = ListOp(expect_ops)

        def cost_evaluation(parameters):
            
            # Create dict associating each parameter with the lists of parameterization values for it
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))    

            # TODO define a multiple sampler, one for each ops, to leverage caching
            # get the sampled output 
            out = []
            for op in expect_ops:
                sampled_expect_op = self._circuit_sampler.convert(op, params=param_bindings)  
                out.append(self.get_probability_from_expected_value(sampled_expect_op.eval()[0]))

            # compute the total cost
            cost = self.process_probability_circuit_output(out)

            # get the internediate results if required
            if self._callback is not None:
                for param_set in parameter_sets:
                    self._eval_count += 1
                    self._callback(self._eval_count, cost, param_set)
            else:
                self._eval_count += 1

            return cost

        return cost_evaluation

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        observable: Optional[Union[LinearSystemObservable, BaseOperator]] = None,
        observable_circuit: Optional[QuantumCircuit] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> Tuple[Union[float, List[float]], Union[float, List[float]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a
             tuple.
        """
        # exit if nothing is provided
        if observable is None and observable_circuit is None:
            return None, None

        # Get the number of qubits
        nb = solution.num_qubits

        # if the observable is given construct post_processing and observable_circuit
        if observable is not None:
            observable_circuit = observable.observable_circuit(nb)
            post_processing = observable.post_processing

            if isinstance(observable, LinearSystemObservable):
                observable = observable.observable(nb)

        is_list = True
        if not isinstance(observable_circuit, list):
            is_list = False
            observable_circuit = [observable_circuit]
            observable = [observable]

        expectations = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(solution.num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))
            expectations.append(~StateFn(obs) @ StateFn(circuit))

        if is_list:
            # execute all in a list op to send circuits in batches
            expectations = ListOp(expectations)
        else:
            expectations = expectations[0]

        # check if an expectation converter is given
        if self._expectation is not None:
            expectations = self._expectation.convert(expectations)
        # if otherwise a backend was specified, try to set the best expectation value
        elif self._circuit_sampler is not None:
            if is_list:
                op = expectations.oplist[0]
            else:
                op = expectations
            self._expectation = ExpectationFactory.build(op, self._circuit_sampler.quantum_instance)

        if self._circuit_sampler is not None:
            expectations = self._circuit_sampler.convert(expectations)

        # evaluate
        expectation_results = expectations.eval()

        # apply post_processing
        result = post_processing(expectation_results, nb, self.scaling)

        return result, expectation_results


    def solve(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                BaseOperator,
                List[LinearSystemObservable],
                List[BaseOperator],
            ]
        ] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> VariationalLinearSolverResult:    
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[List, np.ndarray, QuantumCircuit]): rhs of the linear system
            observable: Optional information to be extracted from the solution.
                Default is `None`.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is `None`.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Raises:
            ValueError: If an invalid combination of observable, observable_circuit and
                post_processing is passed.

        Returns:
            VariationalLinearSolverResult: Result of the optimization and solution vector of the linear system
        """


        # compute the circuits
        circuits = self.construct_circuit(matrix, vector)

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        bounds = _validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient

        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(circuits)

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
            fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
        )   

        # create the solution 
        solution = VariationalLinearSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        solution.cost_function_evals = opt_result.nfev

        # final ansatz
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)

        # observable
        solution.observable = self._calculate_observable(solution.state, observable, 
                                                         observable_circuit, post_processing)

        return solution


def _validate_initial_point(point, ansatz):
    expected_size = ansatz.num_parameters

    # try getting the initial point from the ansatz
    if point is None and hasattr(ansatz, "preferred_init_points"):
        point = ansatz.preferred_init_points
    # if the point is None choose a random initial point

    if point is None:
        # get bounds if ansatz has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(ansatz, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point


def _validate_bounds(ansatz):
    if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
        bounds = ansatz.parameter_bounds
        if len(bounds) != ansatz.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({ansatz.num_parameters})."
            )
    else:
        bounds = [(None, None)] * ansatz.num_parameters

    return bounds

