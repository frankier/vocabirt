from copy import deepcopy
from typing import Optional
from warnings import warn

import numpy
from catsim import cat
from catsim.simulation import Estimator, Selector
from sklearn.linear_model import LogisticRegression


def _fit_log_reg(
    items,
    administered_items,
    response_vector,
    use_discriminations=True,
    # use_guess_slip=True,
    log_reg=None,
):
    if log_reg is None:
        log_reg = LogisticRegression(C=float("inf"))
    X = items[administered_items][:, 1, numpy.newaxis]
    sample_weight = None
    if use_discriminations:
        sample_weight = items[administered_items, 0]
    # if use_guess_slip:
    # response_vector = [
    # slip if resp else guess
    # for resp, (guess, slip) in zip(
    # response_vector, items[administered_items, 2:4]
    # )
    # ]
    log_reg.fit(X, response_vector, sample_weight=sample_weight)
    return log_reg


"""
def _set_log_reg(mean, scale):
    coef = 1 / scale
    _log_reg.intercept_ = -mean * coef
    _log_reg.coef_ = coef
"""


def _log_reg_scale(log_reg):
    return -1 / log_reg.coef_[0, 0]


class LogisticEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __init__(
        self, use_discriminations=True  # , return_scale=False,  # , use_guess_slip=True
    ):
        super().__init__()
        self._use_discriminations = use_discriminations
        # self._use_guess_slip = use_guess_slip
        # self._return_scale = return_scale

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        est_theta: float = None,
        **kwargs,
    ) -> float:
        """Returns the theta value that minimizes the negative log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :param est_theta: a float containing the current estimated proficiency
        :returns: the current :math:`\\hat\\theta`
        """
        items, administered_items, response_vector, est_theta = self._prepare_args(
            return_items=True,
            return_response_vector=True,
            return_est_theta=True,
            index=index,
            items=items,
            administered_items=administered_items,
            response_vector=response_vector,
            est_theta=est_theta,
            **kwargs,
        )
        assert items is not None
        assert administered_items is not None
        assert response_vector is not None
        assert est_theta is not None

        if len(set(response_vector)) == 1:
            return cat.dodd(est_theta, items, response_vector[-1])

        log_reg = _fit_log_reg(
            items,
            administered_items,
            response_vector,
            use_discriminations=self._use_discriminations,
            # use_guess_slip=self._use_guess_slip,
        )
        # y = mx + c, max entropy when y = 0 => x = -c / m
        theta = -log_reg.intercept_[0] / log_reg.coef_[0, 0]
        return theta

        # return theta, _log_reg_scale(log_reg)


def _all_future_scales(
    log_reg, items, administered_items, response_vector, next_choice
):
    res = numpy.zeros((items.shape[0],))
    for item in items[:, 1].argsort():
        log_reg = _fit_log_reg(
            items,
            administered_items + [item],
            response_vector + [next_choice],
            use_discriminations=True,
            log_reg=log_reg,
        )
        scale = abs(_log_reg_scale(log_reg))
        res[item] = scale
    return res


class MinExpectedScaleSelector(Selector):
    """
    Owens 1977,
    """

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        response_vector: list = None,
        **kwargs,
    ) -> Optional[int]:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        items, administered_items, response_vector, est_theta = self._prepare_args(
            return_items=True,
            return_response_vector=True,
            return_est_theta=True,
            index=index,
            items=items,
            administered_items=administered_items,
            response_vector=response_vector,
            est_theta=est_theta,
            **kwargs,
        )

        assert items is not None
        assert administered_items is not None
        assert response_vector is not None
        assert est_theta is not None

        def default():
            # Fall back to max info
            ordered_items = self._sort_by_info(items, est_theta)
            valid_indexes = self._get_non_administered(
                ordered_items, administered_items
            )
            return valid_indexes[0]

        if len(administered_items) > 0 and len(set(response_vector)) >= 2:
            log_reg = LogisticRegression(C=float("inf"), warm_start=True)
            log_reg_before = _fit_log_reg(
                items,
                administered_items,
                response_vector,
                use_discriminations=True,
                log_reg=log_reg,
            )
            if _log_reg_scale(log_reg_before) <= 0:
                return default()

            log_reg.tol = 0.05
            neg_prob, pos_prob = log_reg_before.predict_proba(
                items[:, 1, numpy.newaxis]
            ).T
        else:
            return default()
            # TODO: Can instead use Dodd's like logic to find expected scale even when there is only one class
            # min_theta = min(items[:, 1])
            # max_theta = max(items[:, 1])
            # _set_log_reg(
            # est_theta, min(max_theta - est_theta, est_theta - min_theta)
            # )
        working_log_reg = deepcopy(log_reg)
        false_scales = _all_future_scales(
            working_log_reg, items, administered_items, response_vector, False
        )
        working_log_reg = deepcopy(log_reg)
        true_scales = _all_future_scales(
            working_log_reg, items, administered_items, response_vector, True
        )

        organized_items = [
            x
            for x in numpy.array(
                [
                    pp * ts + np * fs
                    for np, pp, fs, ts in zip(
                        neg_prob, pos_prob, false_scales, true_scales
                    )
                ]
            ).argsort()
            if x not in administered_items
        ]

        if len(organized_items) == 0:
            warn("There are no more items to apply.")
            return None

        return organized_items[0]
