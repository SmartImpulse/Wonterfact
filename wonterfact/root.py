# ----------------------------------------------------------------------------
# Copyright 2020 Smart Impulse SAS, Benoit Fuentes <bf@benoit-fuentes.fr>
#
# This file is part of Wonterfact.
#
# Wonterfact is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Wonterfact is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wonterfact. If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

"""Module for Root class"""

# Python System imports
import time
from functools import cached_property

# Relative imports
from .glob_var_manager import glob
from .core_nodes import _ChildNode
from . import graphviz

# Third-party imports
import numpy as np
import logging
from custom_inherit import doc_inherit


class Root(_ChildNode):
    """
    Class for root node of a wonterfact tree.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        inference_mode : 'EM' of 'VBEM', optional, default 'EM
            Indicates wether the inference algorithm should be the Expectation-
            Maximization (EM) algorithm or the Variational Bayes EM (VBEM)
            algorithm.
        cost_computation_iter : int, optional, default 0
            Indicates the number of algorithm iterations between two evaluations
            of the cost function. 0 means that it is never evaluated. The cost
            function is the sum of two subfunctions, data fitting and constraint
            fitting, whose definitions depend on the used algorithm (EM or VBEM).
            Values of the cost function and subfunctions are recorded in the
            ``Root.cost_record``, ``Root.data_fitting_record`` and
            ``Root.constraints_fitting_record``. Unless simulated annealing is
            used (see ``annealing_proba_init`` parameter), cost function should
            always decrease over the iterations. If not, it means that something
            is wrong in the designed model or buggy in the current version of
            wonterfact package.
        stop_estim_threshold : float, optional, default 0
            When the absolute value of the cost function rate of increase falls
            below this threshold, the parameters estimation algorithm stops.
            Beware that cost function might not be computed at each iteration
            (see ``cost_computation_iter`` parameter). If 0, algorithm never
            stops until it reach the maximum iterations number (see
            ``Root.estim_param`` method).
        update_type : 'regular' or 'parabolic', default 'parabolic'
            If 'regular', regular EM or VBEM algorithm is used. If 'parabolic',
            a parabolic accelation technique with geometric grid search
            is used as described in "Acceleration of the EM algorithm: P-EM
            versus epsilon algorithm", A.F. Berlinet, Ch. Roland, Computational
            Statistics & Data Analysis, Volume 56, Issue 12, December 2012,
            Pages 4122-4137
        parabolic_acc_common_ratio : float, optional, default 1.5
            Common ratio of the geometric grid search of the parabolic
            acceleration (see ``update_type``)
        parabolic_acc_scale_factor : float, optional, default 0.1
            Scale factor for the geometric grid search of the parabolic
            acceleration algorithm (see ``update_type``)
        acceleration_start_iter : int, optional, default 200
            If parabolic acceleration is used, number of regular updates before
            starting it.
        parabolic_forget_distrib_param : float in ]0, 1[, optional, default 0.95
            In order to improve parabolic acceleration, one can try to minimize
            the number of cost function evaluations during grid search phase. To
            do so, it is possible to keep a track of the step distribution given
            the previous optimal step. This distribution could evolve over the
            iteration, and this parameter is a "forget parameter" taking account
            for this possibility. THe closest to 1, the less we forget.
        annealing_proba_init : float, optional, default 0
            Along with the EM or VBEM algorithm, it is possible use a simulated-
            annealing like algorithm in order to better avoid local optima. This
            parameter is the initial probability that parameters are redrawn
            randomly near their current value. If 0, simulated annealing is not
            used.
        annealing_proba_iter_const : float, optional, default 1000
            Parameter for the temperature decrease in the simulated annealing
            method.
        seed : int or None, optional, default None
            Seed for the random generator. If None, seed is random. If integer,
            results should be reproducible, even when simulated annealing is
            used.
        logger : logger object, optional, default None
            Python logger object to write logs. If None, a standard logger is
            used.
        verbose_iter : int, optional, default 0
            Information is given to the logger every ``verbose_iter`` iterations.
        """
        # parse kwargs
        self.inference_mode = kwargs.pop("inference_mode", "EM")
        self.cost_computation_iter = kwargs.pop("cost_computation_iter", 0)
        self.stop_estim_threshold = kwargs.pop("stop_estim_threshold", 0)
        self.update_type = kwargs.pop("update_type", "parabolic")
        self.parabolic_acc_scale_factor = kwargs.pop("parabolic_acc_scale_factor", 0.1)
        self.parabolic_acc_common_ratio = kwargs.pop("parabolic_acc_common_ratio", 1.5)
        self.acceleration_start_iter = kwargs.pop("acceleration_start_iter", 200)
        self.parabolic_forget_distrib_param = kwargs.pop(
            "parabolic_forget_distrib_param", 0.95
        )
        self.annealing_proba_init = kwargs.pop("annealing_proba_init", 0)
        self.annealing_proba_iter_const = kwargs.pop("annealing_proba_iter_const", 1000)
        self.seed = kwargs.pop("seed", None)
        self.logger = kwargs.pop("logger", None)
        self.verbose_iter = kwargs.pop("verbose_iter", 0)

        # fixed attributes
        self.data_fitting_record = []
        self.constraints_fitting_record = []
        self.cost_record = []
        self.current_iter = 0
        self.tic = time.time()
        self.time_duration = []
        self.current_iter_list = []
        self.need_a_bump = False
        super().__init__(**kwargs)

        # logger creation
        self.logger = self.logger or logging.getLogger("wtf")
        self.logger.setLevel(logging.INFO)

        # algorithm state for parabolic acceleration
        nb_step = 50
        step_distrib = np.kron(np.ones((nb_step, 1)), np.linspace(1, 0.99, nb_step))
        step_distrib /= step_distrib.sum(1, keepdims=True)
        self.parab_acc_state = {
            "current_cost": np.inf,
            "standard_update": False,
            "consec_std_update": 0,
            "step_distrib": step_distrib,
            "best_step_list": [],
            "evaluation_number": [],
            "step_min": 0,
            "step_max": nb_step - 1,
            "step_order": np.arange(nb_step),
            "step_to_cost": np.zeros(nb_step),
            "current_step_index": 0,
            "initialization_stage": True,
            "last_best_step": 0,
            "nb_step": nb_step,
            "best_step": 0,
        }

    def tree_traversal(
        self, method_name, mode, method_input=None, iteration_number=None
    ):
        """
        Calls a given method for all the nodes of the tree.

        Parameters
        ----------
        method_name : str
            Method to call. Each node will run this method if it has it.
        mode : 'top-down' or 'bottom-up'
            Whether the methods should be run from leaves to root ('top-down') or
            from root to leaves ('bottom-up').
        method_input : (sequence, dict), optional, default ((), {})
            args and kwargs to give to the method to call
        iteration_number : int or None, optional, default None
            Iteration number that is given to the nodes so they can decide to
            call the given method or not (see ``update_period``, ``update_succ``
            and ``update_offset`` in Nodes docstring)
        """
        method_input = method_input or ((), {})
        if mode == "top-down":
            level_iter = range(0, self.level + 1)
        elif mode == "bottom-up":
            level_iter = range(self.level, -1, -1)
        for level in level_iter:
            for node in self.nodes_by_level[level]:
                self._apply_method_to_node(
                    node, method_name, method_input, iteration_number
                )

    def _apply_method_to_node(self, node, method_name, method_input, iteration_number):
        if hasattr(node, method_name) and node.should_update(iteration_number):
            try:
                node.__getattribute__(method_name)(*method_input[0], **method_input[1])
            except Exception as exception:
                extra_info = "\n Error raised by {} during call of '{}'".format(
                    node, method_name
                )
                raise type(exception)(str(exception) + extra_info)

    @cached_property
    def nodes_by_level(self):
        """
        Get a dictionary of type : {level: [list of nodes]} to get all nodes
        that have a given level
        """
        list_of_all_nodes = self.census()
        nodes_by_level = {}
        for level in range(self.level + 1):
            nodes_by_level[level] = [
                node for node in list_of_all_nodes if node.level == level
            ]
        return nodes_by_level

    @cached_property
    def node_ids_set(self):
        return set(node.name for node in self.census())

    @cached_property
    def nodes_by_id(self):
        list_of_all_nodes = self.census()
        if len(self.node_ids_set) != len(list_of_all_nodes):
            raise ValueError("Several nodes have the same ID")
        return {node.name: node for node in list_of_all_nodes}

    def __getattr__(self, name):
        if name in self.node_ids_set:
            return self.nodes_by_id[name]
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

    def _first_iteration(self, check_model_validity):
        self.tree_traversal(
            "_set_inference_mode",
            mode="top-down",
            method_input=((), dict(mode=self.inference_mode)),
        )
        np.random.seed(self.seed)
        glob.xp.random.seed(self.seed)
        self.tree_traversal("_initialization", mode="top-down")
        if check_model_validity:
            self.tree_traversal("_check_model_validity", mode="top-down")
        self.time_init = time.time()

    def _regular_iteration(self):
        self.time_init = time.time()
        if (
            self.current_iter < self.acceleration_start_iter
            or self.update_type == "regular"
        ):
            self._make_one_step()
            if self.need_a_bump:
                self._launch_bump()
                self.need_a_bump = False
        elif self.update_type == "parabolic":
            self._make_one_parabolic_step()
            # we just wait for the next standard EM or VBEM update to come
            # before launching a bump (simulated annealing algorithm)
            if self.need_a_bump and self.parab_acc_state["standard_update"]:
                self._launch_bump()
                self.need_a_bump = False
        else:
            raise ValueError("Unknown `update_type`")

    def estimate_param(self, n_iter, check_model_validity=True):
        """
        Run parameters estimation algorithm.

        Parameters
        ----------
        n_iter : int
            Indicates the number of iterations to perform. Algorithm could stop
            earlier if convergence is reached (see ``stop_estim_threshold``
            parameter in Root.__init__ docstring).

        Notes
        -----
        It allows stop and go mode, meaning that, assuming that convergence will
        not be reached, calling this method twice with a given ``n_iter=n`` is
        equivalent to calling it once with ``n_iter=2 * n``.
        """
        max_iter = self.current_iter + n_iter

        try:
            for __ in range(n_iter):
                if self.current_iter == 0:
                    self._first_iteration(check_model_validity=check_model_validity)
                    if (
                        self.update_type == "parabolic"
                        and self.inference_mode == "VBEM"
                    ):
                        raise NotImplementedError(
                            "Parabolic acceleration is not yet implemented in"
                            "'VBEM' mode"
                        )
                else:
                    self._regular_iteration()
                self.current_iter += 1

                if (
                    self.cost_computation_iter
                    and self.current_iter % self.cost_computation_iter == 0
                ):
                    self._record_cost_values()

                if self.verbose_iter and self.current_iter % self.verbose_iter == 0:
                    self.verbose(max_iter)

                self.need_a_bump = self._draw_need_a_dump(self.current_iter)
                if self._stop_condition():
                    break

        except KeyboardInterrupt:
            self.tree_traversal(
                "_update_tensor",
                mode="top-down",
                method_input=((), {"update_type": "no_update_for_leaves"}),
                iteration_number=self.current_iter,
            )
            self.logger.info("Parameter estimation stopped by user.")

    def _stop_condition(self):
        stop = len(self.cost_record) >= 2 and (
            abs(self.cost_record[-1] - self.cost_record[-2])
            < self.stop_estim_threshold * abs(self.cost_record[-2])
        )
        return stop

    def _make_one_step(self):
        self.tree_traversal(
            "compute_tensor_update",
            mode="bottom-up",
            iteration_number=self.current_iter,
        )
        self.tree_traversal(
            "_update_tensor", mode="top-down", iteration_number=self.current_iter
        )
        if self.inference_mode == "VBEM":
            #  TODO: also update prior_rate
            self.tree_traversal(
                "_prior_alpha_update",
                mode="bottom-up",
                iteration_number=self.current_iter,
            )

    def _draw_need_a_dump(self, num_iter):
        ran = np.random.rand()
        if ran < self.annealing_proba_init * np.exp(
            -num_iter / self.annealing_proba_iter_const
        ):
            return True
        return False

    def _eval_parabolic_step(self, step):
        update_param = self._give_update_param(step)
        method_kwarg = {"update_type": "parabolic", "update_param": update_param}
        self.tree_traversal(
            "_update_tensor",
            mode="top-down",
            method_input=((), method_kwarg),
            iteration_number=self.current_iter,
        )
        data_fitting = self._get_data_fitting()
        contraints_fitting = self._get_total_contraints_fitting()
        new_cost = data_fitting + contraints_fitting
        self.parab_acc_state["evaluation_number"][-1] += 1

        return new_cost

    def _make_one_parabolic_step_aux1(self):
        self._make_one_step()
        self.tree_traversal(
            "_update_past_tensors", mode="top-down", iteration_number=self.current_iter
        )
        if self.current_iter == self.acceleration_start_iter + 2:
            self.parab_acc_state["current_cost"] = self.get_cost_func()
            self.parab_acc_state["standard_update"] = False
            self.parab_acc_state["initialization_stage"] = True

    def _make_one_parabolic_step_aux2(self):
        self._make_one_step()
        self.parab_acc_state["consec_std_update"] = (
            self.parab_acc_state["consec_std_update"] + 1
        ) % 2
        if self.parab_acc_state["consec_std_update"] == 0:
            self.tree_traversal(
                "_update_past_tensors",
                mode="top-down",
                iteration_number=self.current_iter,
            )
            current_cost = self.get_cost_func()
            self.parab_acc_state["current_cost"] = current_cost
            self.parab_acc_state["standard_update"] = False
            self.parab_acc_state["initialization_stage"] = True

    def _make_one_parabolic_step_aux3(self):
        if self.parab_acc_state["initialization_stage"]:
            step_order = np.argsort(
                self.parab_acc_state["step_distrib"][
                    self.parab_acc_state["last_best_step"]
                ]
            )[::-1]
            step_order = np.array(
                [
                    elem
                    for step in step_order
                    for elem in range(step - 1, step + 2)
                    if 0 <= elem < self.parab_acc_state["nb_step"]
                ]
            )
            _, idx = np.unique(step_order, return_index=True)
            self.parab_acc_state["step_order"][:] = step_order[np.sort(idx)]
            self.parab_acc_state["step_to_cost"][...] = np.nan
            self.parab_acc_state["step_to_cost"][0] = self.parab_acc_state[
                "current_cost"
            ]
            self.parab_acc_state["current_step_index"] = 0
            self.parab_acc_state["step_min"] = 0
            self.parab_acc_state["step_max"] = self.parab_acc_state["nb_step"] - 1
            self.parab_acc_state["best_step"] = 0
            self.parab_acc_state["initialization_stage"] = False
            self.parab_acc_state["evaluation_number"].append(0)

        found_a_better_solution = False
        while not found_a_better_solution:
            step = self.parab_acc_state["step_order"][
                self.parab_acc_state["current_step_index"]
            ]
            if (
                self.parab_acc_state["step_min"]
                <= step
                <= self.parab_acc_state["step_max"]
            ):
                if step > 0:
                    self.parab_acc_state["step_to_cost"][
                        step
                    ] = self._eval_parabolic_step(step)

                if (
                    self.parab_acc_state["step_to_cost"][step]
                    >= self.parab_acc_state["current_cost"]
                ):
                    if step - 1 == self.parab_acc_state["best_step"]:
                        self.parab_acc_state["step_max"] = self.parab_acc_state[
                            "best_step"
                        ]
                    elif step + 1 == self.parab_acc_state["best_step"]:
                        self.parab_acc_state["step_min"] = self.parab_acc_state[
                            "best_step"
                        ]
                    elif step > self.parab_acc_state["best_step"]:
                        self.parab_acc_state["step_max"] = step
                    else:
                        self.parab_acc_state["step_min"] = step
                else:
                    found_a_better_solution = True
                    self.parab_acc_state["best_step"] = step
                    self.parab_acc_state["current_cost"] = self.parab_acc_state[
                        "step_to_cost"
                    ][step]
                    if step > 0 and not np.isnan(
                        self.parab_acc_state["step_to_cost"][step - 1]
                    ):
                        if (
                            self.parab_acc_state["step_to_cost"][step - 1]
                            >= self.parab_acc_state["step_to_cost"][step]
                        ):
                            self.parab_acc_state["step_min"] = step
                        else:
                            self.parab_acc_state["step_max"] = step - 1
                    if step < self.parab_acc_state["nb_step"] - 1 and not np.isnan(
                        self.parab_acc_state["step_to_cost"][step + 1]
                    ):
                        if (
                            self.parab_acc_state["step_to_cost"][step + 1]
                            >= self.parab_acc_state["step_to_cost"][step]
                        ):
                            self.parab_acc_state["step_max"] = step
                        else:
                            self.parab_acc_state["step_min"] = step + 1

                if self.parab_acc_state["step_min"] == self.parab_acc_state["step_max"]:
                    best_step = self.parab_acc_state["step_min"]
                    if step != best_step:
                        update_param = self._give_update_param(best_step)
                        method_kwarg = {
                            "update_type": "parabolic",
                            "update_param": update_param,
                        }
                        self.tree_traversal(
                            "_update_tensor",
                            mode="top-down",
                            method_input=((), method_kwarg),
                            iteration_number=self.current_iter,
                        )
                    self.parab_acc_state["step_distrib"][
                        self.parab_acc_state["last_best_step"]
                    ] *= self.parabolic_forget_distrib_param
                    self.parab_acc_state["step_distrib"][
                        self.parab_acc_state["last_best_step"], best_step
                    ] += (1 - self.parabolic_forget_distrib_param)
                    self.parab_acc_state["best_step_list"].append(best_step)
                    self.parab_acc_state["last_best_step"] = best_step
                    self.parab_acc_state["initialization_stage"] = True
                    self.parab_acc_state["standard_update"] = True
                    if not found_a_better_solution:
                        self._make_one_parabolic_step_aux2()
                        found_a_better_solution = True

            self.parab_acc_state["current_step_index"] += 1

    def _make_one_parabolic_step(self):
        if self.current_iter < self.acceleration_start_iter + 3:
            self._make_one_parabolic_step_aux1()
        elif self.parab_acc_state["standard_update"]:
            self._make_one_parabolic_step_aux2()
        else:
            self._make_one_parabolic_step_aux3()

    def _give_update_param(self, step_number):
        if step_number == 0:
            return 1
        else:
            return (
                1
                + (self.parabolic_acc_common_ratio ** (step_number - 1))
                * self.parabolic_acc_scale_factor
            )

    def _launch_bump(self):
        self.logger.info("Launching annealing")
        self.tree_traversal(
            "_bump", mode="top-down", iteration_number=self.current_iter
        )
        if self.update_type == "parabolic":
            # self.acceleration_start_iter = self.current_iter
            self.parab_acc_state.update(
                {"standard_update": True, "consec_std_update": 0}
            )
        self.need_a_bump = False

    def _record_cost_values(
        self, data_fitting=None, contraints_fitting=None, time_val=None
    ):
        data_fitting = data_fitting or self._get_data_fitting()
        contraints_fitting = contraints_fitting or self._get_total_contraints_fitting()
        time_val = time_val or time.time()
        self.data_fitting_record.append(data_fitting)
        self.constraints_fitting_record.append(contraints_fitting)
        self.cost_record.append(data_fitting + contraints_fitting)
        time_ref = self.time_init if not self.time_duration else self.time_duration[-1]
        time_delta = time_val - self.time_init
        self.time_duration.append(time_ref + time_delta)
        self.current_iter_list.append(self.current_iter)

    def _get_total_contraints_fitting(self):
        """
        Compute minus the sum of all leaves prior values
        """
        return -sum([leaf._prior_value() for leaf in self.nodes_by_level[0]])

    def _get_data_fitting(self):
        parent_fit = 0
        for parent in self.list_of_parents:
            parent_fit += parent._get_data_fitting()
        return parent_fit

    def get_cost_func(self):
        """
        Returns the current value of the cost function
        """
        return self._get_total_contraints_fitting() + self._get_data_fitting()

    def verbose(self, max_iter=None):
        """
        Log info into the logger.
        """
        max_iter = max_iter or self.current_iter
        toc = time.time()
        data_fit = (
            np.nan if not self.data_fitting_record else self.data_fitting_record[-1]
        )
        const_fit = (
            np.nan
            if not self.constraints_fitting_record
            else self.constraints_fitting_record[-1]
        )
        tot = np.nan if not self.cost_record else self.cost_record[-1]
        try:
            data_fit = data_fit.item()
        except:
            pass
        try:
            const_fit = const_fit.item()
        except:
            pass
        try:
            tot = tot.item()
        except:
            pass
        str_verbose = []
        str_verbose.append("it:" + "{:4.0f}/{}".format(self.current_iter, max_iter))
        str_verbose.append("df:" + "{:.3e}".format(data_fit))
        str_verbose.append("cf:" + "{:.3e}".format(const_fit))
        str_verbose.append("tot:" + "{:.3e}".format(tot))
        str_verbose.append("dur:" + "{:2.1f}".format(toc - self.tic))
        self.logger.info("|".join(str_verbose))
        self.tic = toc

    def draw_tree(
        self,
        fileformat=None,
        filename=None,
        legend_dict=None,
        prior_nodes=False,
        view=True,
        show_node_names=False,
    ):
        """
        Draw the tree model with Graphviz (you need to install it and make sure
        that the directory containing the dot executable is on your systems’
        path.)
        Parameters
        ----------
        filename : str, pathlib.Path or None, default None
            Filename to which the graph is saved. If None, the `name` attribute
            of input `tree` is used. If an extension is provided (.svg or .pdf),
            the corresponding format will be used as fileformat.
        fileformat : 'svg', 'pdf', 'png', ..., optional, default 'pdf'
            Rendering output format, if not provided in `filename`'s
            extension. By default, 'pdf' is used.
        legend_dict : dict or None, optional, default None
            A dictionary in the form of `{index_id: idx_dict}` where idx_dict is
            in the form of `{'letter': one_letter_plus_sub_as_str, 'description':
            description_as_str}`. `one_letter_plus_sub_as_str` can either be a
            single letter like 'r' of a letter and a subscript between brackets
            like 'r_{1}'.
        view : bool, default True
            Whether the rendered result is opened is viewer or not
        """
        return graphviz._draw_tree(
            self,
            fileformat=fileformat,
            filename=filename,
            legend_dict=legend_dict,
            prior_nodes=prior_nodes,
            view=view,
            show_node_names=show_node_names,
        )

