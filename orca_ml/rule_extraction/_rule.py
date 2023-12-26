from enum import Enum

import numpy as np
import numexpr as ne
from pandas import DataFrame
from sklearn.utils import check_array


def _get_context(X, feature_names):
    """Make context for code evaluation.

    Parameters
    ------------
    X : array of shape (n_samples, n_features)
        DataFrame should be converted earlier.
    feature_names : array-like of shape (n_features, )
        feature names

    Returns
    ------------
    ctx : dict, evaluation context"""
    return {name: X[:, i] for i, name in enumerate(feature_names)}


def _apply_expr_on_array(expr, X, feature_names):
    ctx = _get_context(X, feature_names)
    return ne.evaluate(expr, local_dict=ctx)


class RuleState(str. Enum):
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(RuntimeError):
    pass


class RuleUnAppliedError(RuleStateError):
    pass


# 中級操作符
op_dict = {"GT": ">", "LT": "<", "EQ": "==", "ADD": "+", "GE": ">=", "LE": "<=", "SUBTRACT": "-", "MULTIPLY": "*", "DIVIDE": "/", "OR": "|", "AND": "&"}
# 数据类型 int, float-->float, 目前不支持 str
value_type_dict = {"int": float, "float": float, "string": str, "bool": bool}
# if_part, then_part, else_part
part_dict = ["if", "then", "else"]


# max_index: 数据的列数     feature_list: 列的名称
def json2expr(data, max_index, feature_list):
    if data.keys()._contains_("operator"):
        op = data.get("operator")
        params = data.get("params")
        if op == "FEATURE_INDEX": # 取变量，一个值{判断变量，索引是否正常}
            feature = params[0].get("feature")
            if params[0].get("index") >= max_index: # json中的索引异常: index >= 数据列数
                raise ValueError("index error")
            if feature not in feature_list: #变量异常:变量名不在数据的列名中
                raise ValueError("{} do not belong to the data ".format(feature))
            return feature
        elif op in op_dict: # 两个值，递归
            value_list = [json2expr(params[0], max_index, feature_list), json2expr(params[1], max_index, feature_list)]
            return "(" + str(value_list[0]) + op_dict[op] + str(value_list[1]) + ")"
        else: # op 不在op_dict报错
            raise TypeError("The operator: {} is invalid".format(op))
        
    if data.keys().__contains__("value"):
        value_type = data.get("value_type")
        value = data["value"]
        # 对取到值的类型做转换， 不在类型字典中的值报错
        if not value_type_dict.get(value_type):
            raise ValueError("Data type error!")
        return value_type_dict.get(value_type)(value)


class Rule:
    """Executable rule for decision making.

    Parameters
    ------------
    expr : str or dict
        The rule expression.

    Examples
    ------------
    description in readme.md
    """
    def __init__(self, expr): # expr 既可以传递字符串，也可以传递dict
        self._state = RuleState.INITIALIZED
        self.expr = expr

    def __str__(self):
        return f"Rule({repr(self.expr)})"
    
    def __repr__(self):
        return f"Rule({repr(self.expr)})"
    
    def predict(self, X, part): # dict预测对应part_dict 、字符串表达式对应"“、其他情况报错
        """Apply rule on the specific data.

        Parameters
        ------------
        X : array-like of shape (n_samples, n_features)
            Data used for applying rules.

        Returns
        ------------
        y_pred : array-like of shape (n_samples, )
            Boolean column represents the rule predicting result.
        """
        if not isinstance(X, DataFrame):
            raise ValueError("Rule can only predict on DataFrame.")
        if self._state == RuleState.INITIALIZED:
            feature_names = (X.columns.values).tolist() # 取数据的列名
            X = check_array(X, dtype=None, ensure_2d=True)
            if isinstance(self.expr, dict): # dict部分
                if part not in part_dict:
                    raise TypeError("Part : {} not in ['if','then','else']".format(part))
                if not self.expr[part]: #没有返回值的情况[]
                    return list()
                dict2expr = json2expr(self.expr[part], X.shape[1], feature_names)
                if not isinstance(dict2expr, str): #返回Value (类型已经做过转换),对其扩充-->fvalue7*Len(X)l
                    result = [dict2expr] * len(X)
                else: # 表达式在进行计算
                    result = _apply_expr_on_array(dict2expr, X, feature_names)
                    result = result.tolist()
                    if not isinstance(result, list): # result只有一个数值时，对其扩充-->[value]*len(X)
                        result = [result] * len(X)
            elif isinstance(self.expr, str): # 字符串表达式部分
                if part != "":
                    raise TypeError('The part of the expression must be ""')
                result = _apply_expr_on_array(self.expr, X, feature_names)
            else:
                raise TypeError("Rule currently only supports dict and expression")
            self.result_ = result
        else:
            result = self.result_
        return result

    def result(self):
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("Invoke `predict` to make a rule applied.")
        return self.result_
    
    def __eq__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        res = self.expr == other.expr
        if self._state == RuleState.INITIALIZED:
            return res
        return res and np.all(self.result() == other.result())
    
    # rule combinations
    def __or__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str):
            r = Rule(f"({self.expr}) | ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_or(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict):
            self.new_dict = {} # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " || "+ str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "OR"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"]. append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"): # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"): # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else: # 两条规则的then都存在
                if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "OR"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
        self.new_dict["then"] = then_part

        # else_part
        else_part = {} # self.else或者other.else存在为空的情况
        if not self.expr.get("else") and not other.expr.get("else"):
            else_part = {}
        elif not self.expr.get("else"): # 一条规则的then存在
            else_part = other.expr.get("else")
        elif not other.expr.get("else"):
            else_part = self.expr.get("else")
        else:
            if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                raise TypeError("两个规则else part类型要一致")
            if self.expr.get("then").get("value_type") != "bool":
                raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
            else_part["value_type"] = "bool"
            else_part["operator"] = "OR"
            else_part["params"] = list()
            else_part["params"].append(self.expr.get("else"))
            else_part["params"].append(other.expr.get("else"))
        self.new_dict["else"] = else_part
        r = Rule(self.new_dict)
        return r

    def __and__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str): # 表达式
            r = Rule(f"({self.expr}) & ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_and(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict): # dict
            self.new_dict = {} # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " && "+ str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "AND"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"].append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"): # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"):  # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else: # 两条规则的then都存在
                if self.expr["then"].get("value_type") != other.expr["then"].get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "AND"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
            self.new_dict["then"] = then_part

            # else_part
            else_part = {} # self.else 或者other.else 存在为空的情况
            if not self.expr.get("else") and not other.expr.get("else"):
                else_part = {}
            elif not self.expr.get("else"):# 一条规则的then存在
                else_part = other.expr.get("else")
            elif not other.expr.get("else"):
                else_part = self.expr.get("else")
            else:
                if self.expr.get("else").get("value_type") != other.expr.get("else").get("value_type"):
                    raise TypeError("两个规则else_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                else_part["value_type"] = "bool"
                else_part["operator"] = "AND"
                else_part["params"] = list()
                else_part["params"].append(self.expr.get("else"))
                else_part["params"].append(other.expr.get("else"))
            self.new_dict["else"] = else_part
            r = Rule(self.new_dict)
            return r
    
    def __xor__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) ^ ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_xor(self.result(), other.result())
        r._state = RuleState.APPLIED
        return r
    
    def __mul__(self, other):
        return self._or_(other)
    
    def __invert__(self):
        r = Rule(f"~({self.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_not(self.result())
        r._state = RuleState.APPLIED
        return r
