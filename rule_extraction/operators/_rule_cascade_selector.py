class RuleCascadeSelector:
    def __init__(self, cascade, op=['and', 'or']):
        self.cascade = cascade
        self.op = op

    def select(self, R):
        op = self._op_to_str()
        mask = []
        for rule in R:
            exp = str(rule)
            cnt = 1
            for ch in exp:
                if ch in op:
                    cnt += 1
                else:
                    continue

            if cnt == self.cascade:
                mask.append( True)
            else:
                mask.append(False)
        
        return [rule for rule, selected in zip(R, mask) if selected]
    
    def _op_to_str(self):
        res = []
        for oper in self.op:
            if oper == 'and':
                res.append('&')
            elif oper == 'or':
                res.append('|')
            elif oper == 'xor':
                res.append('^')
        return res
