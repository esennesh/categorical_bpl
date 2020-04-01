from adt import adt, Case
import torch
import uuid

@adt
class FirstOrderType:
    TOPT: Case
    TENSORT: Case[torch.dtype, torch.Size]
    VART: Case[str]
    ARROWT: Case["FirstOrderType", "FirstOrderType"]

    def _pretty(self, parenthesize=False):
        result = self.match(
            TENSORT=lambda dtype, size: '%s[%s]' % (dtype, size),
            vart=lambda name: name,
            arrowt=lambda l, r: '%s -> %s' % (l._pretty(True), r._pretty())
        )
        if parenthesize and self._key == FirstOrderType._Key.ARROWT:
            result = '(%s)' % result
        return result

    def __str__(self):
        return self._pretty(False)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

def unique_identifier():
    return uuid.uuid4().hex[:7]

def unique_vart():
    return FirstOrderType.VART(unique_identifier())

UNIFICATION_EXCEPTION_MSG = 'Could not unify %s with %s'
SUBSTITUTION_EXCEPTION_MSG = 'to substitute for %s'

class UnificationException(Exception):
    def __init__(self, x, y, k=None):
        self.key = k
        self.vals = (x, y)
        if k:
            msg = UNIFICATION_EXCEPTION_MSG + ' ' + SUBSTITUTION_EXCEPTION_MSG
            self.message = msg % (x, y, k)
        else:
            self.message = UNIFICATION_EXCEPTION_MSG % (x, y)

def try_unify(a, b, subst={}):
    if a == b:
        return a, {}
    if a._key == FirstOrderType._Key.VART:
        return b, {a.vart(): b}
    if b._key == FirstOrderType._Key.VART:
        return a, {b.vart(): a}
    if a._key == FirstOrderType._Key.ARROWT and\
       b._key == FirstOrderType._Key.ARROWT:
        (la, ra) = a.arrowt()
        (lb, rb) = b.arrowt()
        l, lsub = try_unify(la, lb)
        r, rsub = try_unify(ra, rb)
        for k in {**lsub, **rsub}.keys():
            if k in lsub and k in rsub:
                _, sub = try_unify(lsub[k], rsub[k])
                subst.update(sub)
        subst.update(lsub)
        subst.update(rsub)
        return FirstOrderType.ARROWT(l, r), subst
    raise UnificationException(a, b)

def unify(a, b, substitution={}):
    try:
        result, substitution = try_unify(a, b, substitution)
        return substitute(result, substitution)
    except UnificationException:
        return None

def unifier(a, b, substitution={}):
    try:
        _, substitution = try_unify(a, b, substitution)
        return substitution
    except UnificationException:
        return None

def substitute(t, sub):
    return t.match(
        tensort=FirstOrderType.TENSORT,
        vart=lambda m: sub[m] if m in sub else FirstOrderType.VART(m),
        arrowt=lambda l, r: FirstOrderType.ARROWT(substitute(l, sub),
                                              substitute(r, sub))
    )

def fold_arrow(ts):
    if len(ts) == 1:
        return ts[-1]
    return fold_arrow(ts[:-2] + [FirstOrderType.ARROWT(ts[-2], ts[-1])])

def unfold_arrow(arrow):
    return arrow.match(
        tensort=lambda dtype, size: [FirstOrderType.TENSORT(dtype, size)],
        vart=lambda v: [FirstOrderType.VART(v)],
        arrowt=lambda l, r: [l] + unfold_arrow(r)
    )
