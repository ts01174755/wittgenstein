# wittgenstein

_And is there not also the case where we play and--make up the rules as we go along?  
  -Ludwig Wittgenstein_

![the duck-rabbit](https://github.com/imoscovitz/wittgenstein/blob/master/duck-rabbit.jpg)

## Summary

This package implements two iterative coverage-based ruleset algorithms: IREP and RIPPERk.

Performance is similar to sklearn's DecisionTree CART implementation (see [Performance Tests](https://github.com/imoscovitz/ruleset/blob/master/Performance%20Tests.ipynb)).

For explanation of the algorithms, see my article in _Towards Data Science_, or the papers below, under [Useful References](https://github.com/imoscovitz/wittgenstein#useful-references).

## Installation

To install, use
```bash
$ pip install wittgenstein
```

To uninstall, use
```bash
$ pip uninstall wittgenstein
```

## Requirements
- pandas
- numpy
- python version>=3.6

## Usage
Usage syntax is similar to sklearn's.

### Training

Once you have loaded and split your data...
```python
>>> import pandas as pd
>>> df = pd.read_csv(dataset_filename)
>>> from sklearn.model_selection import train_test_split # Or any other mechanism you want to use for data partitioning
>>> train, test = train_test_split(df, test_size=.33)
```
Use the `fit` method to train a `RIPPER` or `IREP` classifier:

```python
>>> import wittgenstein as lw
>>> ripper_clf = lw.RIPPER() # Or irep_clf = lw.IREP() to build a model using IREP
>>> ripper_clf.fit(train, class_feat='Party') # Or pass X and y data to .fit
>>> ripper_clf
<RIPPER with fit ruleset (k=2, prune_size=0.33, dl_allowance=64)> # Hyperparameter details available in the docstrings and TDS article below
```

Access the underlying trained model with the `ruleset_` attribute, or output it with `out_model()`. A ruleset is a disjunction of conjunctions -- 'V' represents 'or'; '^' represents 'and'.

In other words, the model predicts positive class if any of the inner-nested condition-combinations are all true:
```python
>>> ripper_clf.ruleset_
<Ruleset [physician-fee-freeze=n] V [synfuels-corporation-cutback=y^adoption-of-the-budget-resolution=y^anti-satellite-test-ban=n]>
```

`IREP` models tend be higher bias, `RIPPER`'s higher variance.

### ILP with RIPPERk
 輸入基於 RIPPERk 的 ILP package
 ```python
 >>> from wittgenstein import Deduce as RIPPER_D
 ```
 
 Define the new variables and data ranges that can be inferred by ILP.
 ```python
 >>> RD = RIPPER_D.RIPPER_Deduce(df)
 ```
 
 Parse the rule set of RIPPERk and save it as a "rule tree" with FP_Tree data structure.
 ```python
 >>> np_rlist = RD.ruleset_parser(ripper_Base.ruleset_)
 >>> rstree = RD.ruleset_tree(np_rlist)
 ```
 
 Reasoning based on inverse resolution.
 ```python
 >>> RD.rule_Deduce(df,rstree,nor_form='short')
 ```
 parameter:
 - nor_form: 
 "normal" : defalut，The new variables are the disjunction of conjunctions -- 'V' represents 'or'; '^' represents 'and'.
 "short" : The new variables are presented by < var_i, i=0,1,2,...,n >.
 
 Select feature by infomation gain。
 ```python
 train, test = train_test_split(df, test_size=.33)
 df,max_feature_score = RD.feature_select(df,test,target_class='open_flag',n=col_n,round_n=4)
 ```
 parameter：
 - n: Choose the new variables in the top n in infomation.
 - round_n: defalut 4, round the infomation gain to the 'round_n'.
 

### Scoring
To score our trained model, use the `score` function:
```python
>>> X_test = test.drop(class_feat, axis=1)
>>> y_test = test[class_feat]
>>> ripper_clf.score(test_X, test_y)
0.9985686906328078
```

Default scoring metric is accuracy. You can pass in alternate scoring functions, including those available through sklearn:
```python
>>> from sklearn.metrics import precision_score, recall_score
>>> precision = clf.score(X_test, y_test, precision_score)
>>> recall = clf.score(X_test, y_test, recall_score)
>>> print(f'precision: {precision} recall: {recall}')
precision: 0.9914..., recall: 0.9953...
```

### Model selection
wittgenstein is compatible with sklearn model_selection tools such as `cross_val_score` and `GridSearchCV`, as well
as ensemblers like `StackingClassifier`.

Cross validation:
```python
>>> # First dummify your categorical features and booleanize your class values to make sklearn happy
>>> X_train = pd.get_dummies(X_train, columns=X_train.select_dtypes('object').columns)
>>> y_train = y_train.map(lambda x: 1 if x=='democrat' else 0)
>>> cross_val_score(ripper, X_train, y_train)
```

Grid search:
```python
>>> param_grid = {"prune_size": [0.33, 0.5], "k": [1, 2]}
>>> grid = GridSearchCV(estimator=ripper, param_grid=param_grid)
>>> grid.fit(X_train, y_train)
```

Ensemble:
```python
>>> tree = DecisionTreeClassifier(random_state=42)
>>> nb = GaussianNB(random_state=42)
>>> estimators = [("rip", ripper_clf), ("tree", tree), ("nb", nb)]
>>> ensemble_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
>>> ensemble_clf.fit(X_train, y_train)
```

### Prediction
To perform predictions, use `predict`:
```python
>>> ripper_clf.predict(new_data)[:5]
[True, True, False, True, False]
```

Predict class probabilities with `predict_proba`:
```python
>>> ripper_clf.predict_proba(test)
# Pairs of negative and positive class probabilities
array([[0.01212121, 0.98787879],
       [0.01212121, 0.98787879],
       [0.77777778, 0.22222222],
       [0.2       , 0.8       ],
       ...
```

We can also ask our model to tell us why it made each positive prediction using `give_reasons`:
```python
>>> ripper_clf.predict(new_data[:5], give_reasons=True)
([True, True, False, True, True]
[<Rule [physician-fee-freeze=n]>],
[<Rule [physician-fee-freeze=n]>,
  <Rule [synfuels-corporation-cutback=y^adoption-of-the-budget-resolution=y^anti-satellite-test-ban=n]>], # This example met multiple sufficient conditions for a positive prediction
[],
[<Rule object: [physician-fee-freeze=n]>],
[])
```

### Altering models
Sometimes you may wish to specify or modify a model (for instance, to take into account subject matter expertise, to create a baseline for scoring, to make predictions based on your own intuitions -- or perhaps out of curiosity to see how it does).

To specify your own model, use `init_ruleset`:
```python
>>> ripper_clf.init_ruleset('[[physician-fee-freeze=n] V [anti-satellite-test-ban=n^physician-fee-freeze=y]]')
```

To modify a trained model, use `add_rule`, `replace_rule`, `remove_rule`, or `insert_rule`. To alter a model by index, use `replace_rule_at`, etc.
```python
>>> ripper_clf.replace_rule_at(1, '[anti-satellite-test-ban=n]')
>>> ripper_clf.insert_rule(insert_before_rule='[physician-fee-freeze=n]', new_rule='[endorse-compulsory-piracy=y]')
>>> ripper_clf.out_model()
[[endorse-compulsory-piracy=y] V
[physician-fee-freeze=n] V
[anti-satellite-test-ban=n]]
```

## Issues
If you encounter any issues, or if you have feedback or improvement requests for how wittgenstein could be more helpful for you, please post them to [issues](https://github.com/imoscovitz/wittgenstein/issues), and I'll respond.

## Contributing
Contributions are welcome! If you are interested in contributing, let me know at ilan.moscovitz@gmail.com or on [linkedin](https://www.linkedin.com/in/ilan-moscovitz/).

## Useful references
- [My article in _Towards Data Science_ explaining IREP, RIPPER, and wittgenstein](https://towardsdatascience.com/how-to-perform-explainable-machine-learning-classification-without-any-trees-873db4192c68)
- [Furnkrantz-Widmer IREP paper](https://pdfs.semanticscholar.org/f67e/bb7b392f51076899f58c53bf57d5e71e36e9.pdf)
- [Cohen's RIPPER paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)
- [Partial decision trees](https://researchcommons.waikato.ac.nz/bitstream/handle/10289/1047/uow-cs-wp-1998-02.pdf?sequence=1&isAllowed=y)
- [Bayesian Rulesets](https://pdfs.semanticscholar.org/bb51/b3046f6ff607deb218792347cb0e9b0b621a.pdf)
- [C4.5 paper including all the gory details on MDL](https://pdfs.semanticscholar.org/cb94/e3d981a5e1901793c6bfedd93ce9cc07885d.pdf)
- [_Philosophical Investigations_](https://static1.squarespace.com/static/54889e73e4b0a2c1f9891289/t/564b61a4e4b04eca59c4d232/1447780772744/Ludwig.Wittgenstein.-.Philosophical.Investigations.pdf)

## Changelog

#### v0.2.3: 5/21/2020
- Minor bugfixes and optimizations

#### v0.2.0: 5/4/2020
- Algorithmic optimizations to improve training speed (~10x - ~100x)
- Support for training on iterable datatypes besides DataFrames, such as numpy arrays and python lists
- Compatibility with sklearn ensembling metalearners and sklearn model_selection
- `.predict_proba` returns probas in neg, pos order
- Certain parameters (hyperparameters, random_state, etc.) should now be passed into IREP/RIPPER constructors rather than the .fit method.
- Sundry bugfixes
