from adt import adt, Case
import torch
import uuid

@adt
class FirstOrderType:
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
