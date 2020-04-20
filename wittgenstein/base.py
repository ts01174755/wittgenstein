""" Base classes for ruleset classifiers """

import copy
import math

from abc import ABC, abstractmethod
import numpy as np
from numpy import var, mean

import warnings


def _warn(message, category, filename, funcname, warnstack=[]):
    """ warnstack: (optional) list of tuples of filename and function(s) calling the function where warning occurs """
    message = "\n" + message + "\n"
    filename += ".py"
    funcname = " ." + funcname
    if warnstack:
        filename = (
            str(
                [
                    stack_filename + ".py: ." + stack_funcname + " | "
                    for stack_filename, stack_funcname in warnstack
                ]
            )
            .strip("[")
            .strip("]")
            .strip(" ")
            .strip("'")
            + filename
        )
    warnings.showwarning(message, category, filename=filename, lineno=funcname)


class AbstractRulesetClassifier(ABC):

    def __init__(
        self,
        prune_size=0.33,
        n_discretize_bins=10,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
        verbosity=0
    ):
        super().__init__()

        self.VALID_HYPERPARAMETERS = {
            "prune_size",
            "n_discretize_bins",
            "max_rules",
            "max_rule_conds",
            "max_total_conds",
            "random_state",
            "verbosity",
        }
        self.prune_size = prune_size
        self.n_discretize_bins = n_discretize_bins
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.max_total_conds = max_total_conds
        self.random_state = random_state
        self.verbosity = verbosity


class Ruleset:
    """ Base Ruleset model.
        Implements collection of Rules in disjunctive normal form.
    """

    def __init__(self, rules=None):
        if rules is None:
            self.rules = []
        else:
            self.rules = rules
        self.cond_count = 0

    def __str__(self):
        return " V ".join([str(rule) for rule in self.rules])

    def __repr__(self):
        ruleset_str = self.__str__()
        return f"<Ruleset object: {ruleset_str}>"

    def __getitem__(self, index):
        return self.rules[index]

    def __len__(self):
        return len(self.rules)

    def truncstr(self, limit=2, direction="left"):
        """ Return Ruleset string representation limited to a specified number of rules.

            limit: how many rules to return
            direction: which part to return. (valid options: 'left', 'right')
        """
        if len(self.rules) > limit:
            if direction == "left":
                return Ruleset(self.rules[:limit]).__str__() + "..."
            elif direction == "right":
                return "..." + Ruleset(self.rules[-limit:]).__str__()
            else:
                raise ValueError('direction param must be "left" or "right"')
        else:
            return self.__str__()

    def __eq__(self, other):
        # if type(other)!=Ruleset:
        #    raise TypeError(f'{self} __eq__ {other}: a Ruleset can only be compared with another Ruleset')
        for (
            r
        ) in (
            self.rules
        ):  # TODO: Ideally, should implement a hash function--in practice speedup would be insignificant
            if r not in other.rules:
                return False
        for (
            r
        ) in (
            other.rules
        ):  # Check the other way around too. (Can't compare lengths instead b/c there might be duplicate rules.)
            if r not in self.rules:
                return False
        return True

    def out_pretty(self):
        """ Prints Ruleset line-by-line. """
        ruleset_str = (
            str([str(rule) for rule in self.rules])
            .replace(" ", "")
            .replace(",", " V\n")
            .replace("'", "")
        )
        print(ruleset_str)

    def copy(self, n_rules_limit=None):
        """ Returns a deep copy of self.

            n_rules_limit (optional): keep only this many rules from the original.
        """
        result = copy.deepcopy(self)
        if n_rules_limit is not None:
            result.rules = result.rules[:n_rules_limit]
        return result

    def covers(self, df):
        """ Returns instances covered by the Ruleset. """
        allpos, allneg = self._check_allpos_allneg(warn=False)
        if allpos:
            return df
        elif allneg:
            return df.head(0)
        else:
            covered = self.rules[0].covers(df).copy()
            for rule in self.rules[1:]:
                covered = covered.append(rule.covers(df))
            covered = covered.drop_duplicates()
            return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def add(self, rule):
        self.rules.append(rule)

    def count_rules(self):
        """ Returns number of rules in the Ruleset.
            (For ease of use for users who don't make a habit of directly accessing the Ruleset object.)
        """
        return len(self.rules)

    def count_conds(self):
        """ Returns the total number of conditions in the Ruleset. This is a measurement of complexity that's the conceptual equivalent of counting the nodes in a decision tree. """
        return sum([len(r.conds) for r in self.rules])

    def _set_possible_conds(self, pos_df, neg_df):
        """ Stores a list of all possible conds. """

        # Used in Rule::successors so as not to rebuild it each time,
        # and in exceptions_dl calculations because nCr portion of formula already accounts for no replacement.)

        self.possible_conds = []
        for feat in pos_df.columns.values:
            for val in set(pos_df[feat].unique()).intersection(
                set(neg_df[feat].unique())
            ):
                self.possible_conds.append(Cond(feat, val))

    def trim_conds(self, max_total_conds=None):
        """ Reduce the total number of Conds in a Ruleset by removing Rules """
        if max_total_conds is not None:
            while len(self.rules) > 0 and self.count_conds() > max_total_conds:
                self.rules.pop(-1)

    def trimmed_str(iterable, max_items=3):
        return str(iterable[:max_items])[-1] + "..."

    def predict(self, X_df, give_reasons=False, warn=True):
        """ Predict classes of data using a fit Ruleset model.

            args:
                X_df <DataFrame>: examples to make predictions on.

                give_reasons (optional) <bool>: whether to provide reasons for each prediction made.

            returns:
                list of <bool> values corresponding to examples. True indicates positive predicted class; False non-positive class.

                If give_reasons is True, returns a tuple that contains the above list of predictions
                    and a list of the corresponding reasons for each prediction;
                    for each positive prediction, gives a list of all the covering Rules, for negative predictions, an empty list.
        """

        # Issue warning if Ruleset is universal or empty
        self._check_allpos_allneg(warn=warn, warnstack=[("base", "predict")])

        covered_indices = set(self.covers(X_df).index.tolist())
        predictions = [i in covered_indices for i in X_df.index]

        if not give_reasons:
            return predictions
        else:
            reasons = []
            # For each Ruleset-covered example, collect list of every Rule that covers it;
            # for non-covered examples, collect an empty list
            for i, p in zip(X_df.index, predictions):
                example = X_df[X_df.index == i]
                example_reasons = (
                    [rule for rule in self.rules if len(rule.covers(example)) == 1]
                    if p
                    else []
                )
                reasons.append(example_reasons)
            return (predictions, reasons)

    def predict_proba(self, X_df, give_reasons=False):
        """ Predict probabilities for each class using a fit Ruleset model.

                args:
                    X_df <DataFrame>: examples to make predictions on.

                    give_reasons (optional) <bool>: whether to provide reasons for each prediction made.
                    min_samples (optional) <int>: return None for each example proba that lack this many samples
                                                  set to None to ignore. (default=None)

                    give_reasons (optional) <bool>: whether to also return reasons for each prediction made.
                    ret_n (optional) <bool>: whether to also return the number of samples used for calculating each examples proba

                returns:
                    numpy array of values corresponding to each example's classes probabilities, or, if give_reasons or ret_n, a tuple containing proba array and list(s) of desired returns
                    a sample's class probabilities will be None if there are fewer than @param min_samples
        """

        # probas for all negative predictions
        uncovered_proba = weighted_avg_freqs([self.uncovered_class_freqs])
        # uncovered_n = sum(self.uncovered_class_freqs)

        # make predictions
        predictions, covering_rules = self.predict(X_df, give_reasons=True, warn=False)
        # N = []

        # collect probas
        probas = np.empty(shape=(len(predictions), uncovered_proba.shape[0]))
        for i, (p, cr) in enumerate(zip(predictions, covering_rules)):
            # n = sum([sum(rule.class_freqs) for rule in cr]) # if user requests, check to ensure valid sample size
            # if (p==True) and (n < 1 or (min_samples and n < min_samples)):
            #    probas[i, :] = None
            # N.append(n)
            # elif (p==False) and (uncovered_n < 1 or uncovered_n < min_samples):
            #    probas[i, :] = None
            # N.append(n)
            # elif p: # pos prediction

            if not p:  # neg prediction
                probas[i, :] = weighted_avg_freqs([rule.class_freqs for rule in cr])
                # N.append(n)
            elif p:  # pos prediction
                probas[i, :] = uncovered_proba
                # N.append(uncovered_n)
        # return probas (and optional extras)
        result = flagged_return([True, give_reasons], [probas, covering_rules])
        return result

    def _check_allpos_allneg(self, warn=False, warnstack=""):
        """ Return tuple<bool> representing whether a Ruleset is universal (always predicts pos), empty (always predicts neg) """
        allpos = self.rules == [Rule()]
        allneg = self.rules == []
        if allpos and warn:
            warning_str = f"Ruleset is universal. All predictions it makes with method .predict will be positive. It may be untrained or was trained on a dataset split lacking negative examples."
            _warn(
                warning_str,
                RuntimeWarning,
                filename="base",
                funcname="_check_allpos_allneg",
                warnstack=warnstack,
            )
        elif allneg and warn:
            warning_str = f"Ruleset is empty. All predictions it makes with method .predict will be negative. It may be untrained or was trained on a dataset split lacking positive examples."
            _warn(
                warning_str,
                RuntimeWarning,
                filename="base",
                funcname="_check_allpos_allneg",
                warnstack=warnstack,
            )
        return allpos, allneg

    def get_selected_features(self):
        feature_list = []
        feature_set = set()
        for rule in self.rules:
            for cond in rule.conds:
                feature = cond.feature
                if feature not in feature_set:
                    feature_list.append(feature)
                    feature_set.add(feature)
        return feature_list


class Rule:
    """ Class implementing conjunctions of Conds """

    def __init__(self, conds=None):
        if conds is None:
            self.conds = []
        else:
            self.conds = conds

    def __str__(self):
        if not self.conds:
            rule_str = "[True]"
        else:
            rule_str = (
                str([str(cond) for cond in self.conds])
                .replace(",", "^")
                .replace("'", "")
                .replace(" ", "")
            )
        return rule_str

    def __repr__(self):
        return f"<Rule object: {str(self)}>"

    def __add__(self, cond):
        if isinstance(cond, Cond):
            return Rule(self.conds + [cond])
        else:
            raise TypeError(
                f"{self} + {cond}: Rule objects can only conjoin Cond objects."
            )

    def __eq__(self, other):
        # if type(other)!=Rule:
        #    raise TypeError(f'{self} __eq__ {other}: a Rule can only be compared with another rule')
        if len(self.conds) != len(other.conds):
            return False
        return set([str(cond) for cond in self.conds]) == set(
            [str(cond) for cond in other.conds]
        )

    def __hash__(self):
        return hash(str([self.conds]))

    def isempty(self):
        return len(self.conds) == 0

    def covers(self, df):
        """ Returns instances covered by the Rule. """
        covered = df.head(len(df))
        for cond in self.conds:
            covered = cond.covers(covered)
        return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def covered_feats(self):
        """ Returns list of features covered by the Rule """
        return [cond.feature for cond in self.conds]

    #############################################
    ##### Rule::grow/prune helper functions #####
    #############################################

    def successors(self, possible_conds, pos_df, neg_df):
        """ Returns a list of all valid successor rules.

        possible_conds: list of Conds to consider conjoining to create successors.
                        passing None defaults to create this param from pos_df and neg_df --
                        however, if pos_df and neg_df are data subsets, it will only generate possible_conds
                        from their available values.
        """

        if possible_conds is not None:
            successor_conds = [
                cond for cond in possible_conds if cond not in self.conds
            ]
            return [Rule(self.conds + [cond]) for cond in successor_conds]
        else:
            successor_rules = []
            for feat in pos_df.columns.values:
                for val in set(pos_df[feat].unique()).intersection(
                    set(neg_df[feat].unique())
                ):
                    if (
                        feat not in self.covered_feats()
                    ):  # Conds already in Rule and Conds that contradict Rule aren't valid successors / NB Rules are short; this is unlikely to be worth the overhead of cheacking
                        successor_rules.append(self + Cond(feat, val))
            return successor_rules


class Cond:
    """ Class implementing conditional. """

    def __init__(self, feature, val):
        self.feature = feature
        self.val = val

    def __str__(self):
        return f"{self.feature}={self.val}"

    def __repr__(self):
        return f"<Cond object: {self.feature}={self.val}>"

    def __eq__(self, other):
        return self.feature == other.feature and self.val == other.val

    def __hash__(self):
        return hash((self.feature, self.val))

    def covers(self, df):
        """ Returns instances covered by the Cond (i.e. those which are not in contradiction with it). """
        return df[df[self.feature] == self.val]

    def num_covered(self, df):
        return len(self.covers(df))


######################
######## MATH ########
######################


def weighted_avg_freqs(counts):
    """ Return weighted mean proportions of counts in the list

        counts <list<tuple>>
    """
    arr = np.array(counts)
    total = arr.flatten().sum()
    return arr.sum(axis=0) / total if total else arr.sum(axis=0)


def flagged_return(flags, objects):
    """ Returns only objects with corresponding True flags
        Useful when """
    if sum(flags) == 1:
        return objects[0]
    elif sum(flags) > 1:
        return tuple([object for flag, object in zip(flags, objects) if flag])
    else:
        return ()


def rnd(float, places="default"):
    """ places: number of decimal places to round.
                set to 'default': defaults to 1 decimal place if float < 100, otherwise defaults to 0 places
    """
    if places == "default":
        if float < 1:
            places = 2
        elif float < 100:
            places = 1
        else:
            places = 0
    rounded = round(float, places)
    if rounded != int(rounded):
        return rounded
    else:
        return int(rounded)
