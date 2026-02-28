
from __future__ import annotations

import math
from typing import Union

class Value:
    """
    Clase core del script que se encarga de calcular los gradients y construir el
    grafo de computación.
    """

    # por defecto, todos los atributos de un objeto en Python se guardan en un __dict__
    # __dict__ es muy flexible pero su naturaleza dinámica es más costosa a nivel de
    # memoria y velocidad
    # nosotros vamos a tener millones de instancias de Value pero no necesitamos
    # atributos nuevos, como el suguiente: obj.foo = 42 (que __dict__) permitiría
    # por este motivo, vamos a usar __slots__ y Python únicamente creará estos atributos
    # __slots__ elimina la parte "dinámica" de un diccionari
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(
        self, *, data: int, children: tuple = (), local_grads: tuple = ()
    ) -> None:

        # scalar value of this node calculated during forward pass
        self.data = data

        # derivative of the loss w.r.t. this node, calculated in backward pass
        self.grad = 0

        # children of this node in the computation graph
        self._children = children

        # local derivative of this node w.r.t. its children
        self._local_grads = local_grads

    # __ magic methods
    # dunder methods
    # double underscore
    # __add__ permite hacer la suma
    def __add__(self, other: Union[Value, int]) -> Value:
        """Suma self.data con other.data"""

        # check is es de la clase Value y la convertimos en caso contrario
        other = other if isinstance(other, Value) else Value(data=other)

        # devolvemos una instancia de Value, sumando sus data + other.data
        return Value(data = self.data + other.data, children=(self, other), local_grads=(1, 1))

    def __repr__(self):
        _repr = f"""
        Value(
            data={self.data},
            children={self._children},
            local_grads={self._local_grads}
        )
        """

        _repr = f"""
        Value(
            data={self.data},
        )
        """

        return _repr

    def __mul__(self, other: Union[Value, int]) -> Value:
        """Multiplica self.data con other.data"""

        # check is es de la clase Value y la convertimos en caso contrario
        other = other if isinstance(other, Value) else Value(data=other)

        # devolvemos una instancia de Value, multiplicando sus data + other.data

        return Value(data=self.data * other.data, children=(self, other), local_grads=(other.data, self.data))

    def __pow__(self, other: int) -> Value:
        """Potencia self.data con other.data"""

        # devolvemos una instancia de Value, multiplicando sus data + other.data
        # porque?
        # local_grads=(other * self.data ** (other - 1),))

#        print(other)
#        print(self.data)
#        print(other * self.data)
#        print(other * self.data ** (other - 1))

        return Value(data=self.data**other, children=(self,), local_grads=(other * self.data ** (other - 1),))

    def log(self):
        return Value(data = math.log(self.data), children=(self,), local_grads=(1 / self.data,))

    def exp(self):
        return Value(data=math.exp(self.data), children=(self,), local_grads=(math.exp(self.data),))

    def relu(self):
        return Value(data=max(0, self.data), children=(self,), local_grads=(float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        # __add__ self + other
        # __radd__ other + self
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        # división sin más
        return self * other**-1

    def __rtruediv__(self, other):
        # inversa div sin más
        return other * self**-1

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1

        print(topo)
        print(visited)

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

if __name__ == "__main__":

    val1 = Value(data=10.0)
    val2 = Value(data=15.0)

    print("Inputs")
    print(val1, val2)

#    suma_vals = val1 + val2
#    print(suma_vals)
#
#    mult_vals = val1 * val2
#    print(mult_vals)
#
#    pot_vals = val1 ** 15
#    print(pot_vals)
#
#    print(val1.log())
#    print(val1.relu())

#     print(-val1)


#    suma_vals_inv = val2 + val1
#    print(suma_vals_inv)

#    print("resta")
#    resta_vals = val1 - val2
#    print(resta_vals)
#
#    print("resta inv")
#    resta_vals_inv = val2 - val1
#    print(resta_vals_inv)
#
#    print("mult inv")
#    mult_vals_inv = val2 * val1
#    print(mult_vals_inv)

#    true_div_ = val1.__truediv__(val2)
#    print(true_div_)
#
#    true_div2_ = val1 / val2
#    print(true_div2_)

    val1.backward()

