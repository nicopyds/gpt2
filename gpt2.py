"""
Teardown del script de Karpathy donde implementa el algoritmo de GPT2
---------------------------------------------------------------------

El original se puede encontrar en este enlace
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Una visualización muy muy útil para entender GPT2 y el entrenamiento es
esta página web:
https://ko-microgpt.vercel.app/#lesson-1

The most atomic way to train and run inference for a GPT in pure, dependency-free
Python. This file is the complete algorithm. Everything else is just efficiency.

@karpathy
"""

from __future__ import annotations

import math
import os
import random
import urllib.request
from typing import Union

random.seed(42)

VERBOSE = True
DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def get_data() -> list[str]:
    """
    Miramos si tenemos o no datos dentro de carpeta donde tenemos el fichero gpt2.py
    y en caso contrario lo descargamos para poder hacer luego nuestro entrenamiento.

    Los datos se encuentran en este link:

    https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt

    Parameters:
    -----------
    None

    Returns:
    --------
    docs: list[str] with the names found in the input.txt file
    """
    input_path_ = os.path.join(DIR_PATH, "input.txt")

    if not os.path.exists(input_path_):
        url_ = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(url_, "input.txt")
        docs = [line.strip() for line in open(input_path_) if line.strip()]

    else:
        docs = [line.strip() for line in open(input_path_) if line.strip()]

    # random.shuffle se hace inplace
    random.shuffle(docs)

    return docs


# dentro de docs tenemos una lista de nombres
# con este código, hacemos una set de las letras únicas
# docs = ['nico', 'paco']
# ucahrs = ['a', 'c', 'i', 'n', 'o', 'p']
docs = get_data()
uchars = sorted(set("".join(docs)))

# queremos tener un id de un token especial llamado Beginning of Sequence (BOS)
# para ellos calculamos el len de nuestra uchars
# el tamaño de nuestros tokens únicos es: ucars + BOS
BOS = len(uchars)
vocab_size = len(uchars) + 1


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
        self, data: float, children: tuple = (), local_grads: tuple = ()
    ) -> None:

        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __repr__(self):
        """
        Esta función dunder de Python sirve para hacer la "representación"
        del objeto con el que estamos trabajando.

        A nosotros nos ayuda, cuando hacemos un print del objeto, ver sus principales
        atributos y "reproducir" este objeto.

        En caso contrario, cuando hacemos un print, tendríamos la dirección del
        objeto en memoria.
        """
        _repr = f"""
            Value(
                data={self.data},
                grad={self.grad},
                children={self._children},
                local_grads={self._local_grads}
            )
        """
        return _repr

    def __add__(self, other: Union[Value, float]) -> Value:
        """
        Suma self.data con other.data

        Dunder add, le dice a Python, como debe sumar los objetos cuando se
        encuentra con algo parecido a: val1 + val2

        En el método de suma, la derivada local de la suma  es 1 para ambos
        operandos, por eso local_grads=(1, 1)

        f(a,b) = a+b
        𝜕𝑓/𝜕𝑎  = 1
        𝜕𝑓/𝜕𝑏  = 1

        Un cambio en a o b produce el mismo cambio en f.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val3 = val1 + val2

        Outputs:
        Value(
            data=25.0,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=15.0,
                    grad=0,
                    children=(),
                    local_grads=()
                )
            ),
            local_grads=(1,1)
        )
        """
        other = other if isinstance(other, Value) else Value(other)

        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Union[Value, float]) -> Value:
        """
        Multiplica self.data con other.data

        Dunder mul, le dice a Python, como debe multiplicar los objetos cuando se
        encuentra con algo parecido a: val1 * val2

        val3 = val1 * val2

        En el método de multiplicar, la derivada local de la multiplicación es el valor
        del otro operando, por eso local_grads=(other.data, self.data).

        Se han intercambiado los operandos.

        f(a,b) = a*b
        𝜕𝑓/𝜕𝑎  = b
        𝜕𝑓/𝜕𝑏  = a

        La derivada de la multiplicación respecto a un factor es el valor del otro
        factor.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val1 * val2

        Outputs:
        Value(
            data=150.0,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=15.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
            ),
            local_grads=(15.0, 10.0)
        )
        """
        other = other if isinstance(other, Value) else Value(other)

        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: Union[Value, int]) -> Value:
        """
        Potencia self.data con other.data

        Dunder pow, le dice a Python, como debe elevar los objetos cuando se
        encuentra con algo parecido a: val1 ** val2

        f(a)  = a**n, donde n es una constante
        𝜕𝑓/𝜕𝑎 = n*a**n-1

        La derivada de a**3 = 3*a**2
        La derivada de a**4 = 4*a**3
        etc

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val1 ** val2

        Outputs:

        Value(
            data=1000000000000000.0,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
            ),
            local_grads=(1500000000000000.0,)
        )
        """
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        """
        Este método, no es un dunder method, calcula el logaritmo de un valor.

        f(a)  = ln(a)
        ∂f/∂a = 1/a

        La derivada del logaritmo es 1/a.
        Si a es 10, la derivada es 0.1.
        Si a es 100, la derivada es 0.01
        Cuanto más grande es a, más pequeña es la derivada del logaritmo.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10)
        val1.log()

        Outputs:
        Value(
            data=2.302585092994046,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
            ),
            local_grads=(0.1,)
        )
        """

        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        """
        Este método, no es un dunder method, calcula la exponenciación de un valor.

        f(a)  = e**a
        𝜕f/𝜕𝑎 = e**a

        La derivada de la exponenciación es la propia función exponencial.
        Por este motivo se usa tanto en las Redes Neuronales, es muy trivial calcularla.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10)
        val1.exp()

        Es el número irracional `e` ** 10

        Outputs:
        Value(
            data=22026.465794806718,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
            ),
            local_grads=(22026.465794806718,)
        )
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        """
        Función básica de activación.

        Si el valor es negativo, devolvemos cero en caso contrario devolvemos el
        propio valor.

        f(a)=max(0,a) = {a if a>0
                        {0 if a≤0
        𝜕𝑓/𝜕𝑎         = {1 if a > 0
                        {0 if a ≤ 0

        La ReLu actúa de la siguiente manera:
        Si el valor es negativo, la derivada es cero, por lo que no se propaga el
        gradiente.

        Si el valor es positivo, la derivada es 1, por lo que el gradiente se propaga
        sin cambios.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10)
        val1.relu()

        Outputs:
        Value(
           data=10.0,
           grad=0,
           children=(
               Value(
                   data=10.0,
                   grad=0,
                   children=(),
                   local_grads=()
               ),
            ),
           local_grads=(1.0,)
        )

        """
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        """
        Dunder method que permite a Python tratar este tipo de operaciones.
        val1 = Value(data=10)
        -val1

        Utilizará debajo, el método __mul__(other=-1)

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10)
        -val1

        Outputs:
        Value(
            data=-10.0,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=-1,
                    grad=0,
                    children=(),
                    local_grads=()
                )
            ),
            local_grads=(-1, 10.0)
        )
        """
        return self * -1

    def __radd__(self, other):
        """
        Suma other.data con self.data

        El método de __radd__ es reverse add. Cuando el __add__ habitual
        te devuelve un NotImplementedError.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val3 = val2 + val1

        Outputs:
        Value(
            data=25.0,
            grad=0,
            children=(
                Value(
                    data=15.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                )
            ),
            local_grads=(1, 1)
        )
        """

        return self + other

    def __sub__(self, other):
        """
        Operación de resta en Python.
        Para que el lenguaje sepa que hacer cuando se encuentra con
        val1 - val2

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val3 = val1 - val2

        Outputs:
        Value(
            data=-5.0,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=-15.0,
                    grad=0,
                    children=(
                        Value(
                            data=15.0,
                            grad=0,
                            children=(),
                            local_grads=()
                        ),
                        Value(
                            data=-1,
                            grad=0,
                            children=(),
                            local_grads=()
                        )
                    ),
                    local_grads=(-1, 15.0)
                )
            ),
            local_grads=(1, 1)
        )
        """

        return self + (-other)

    def __rsub__(self, other):
        """
        Operación de resta inversa en Python.

        El método de __rsub__ es reverse. Cuando el __sub__ habitual
        te devuelve un NotImplementedError.

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val3 = val2 - val1

        Outputs:
        Value(
            data=5.0,
            grad=0,
            children=(
                Value(
                    data=15.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=-10.0,
                    grad=0,
                    children=(
                        Value(
                            data=10.0,
                            grad=0,
                            children=(),
                            local_grads=()
                        ),
                        Value(
                            data=-1,
                            grad=0,
                            children=(),
                            local_grads=()
                        )
                    ),
                    local_grads=(-1, 10.0)
                )
            ),
            local_grads=(1, 1)
        )
        """
        return other + (-self)

    def __rmul__(self, other):
        """
        Multiplica other.data con self.data

        Dunder mul, le dice a Python, como debe multiplicar los objetos cuando se
        encuentra con algo parecido a: val2 * val1

        val3 = val2 * val1

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val2 * val1

        Outputs:
        Value(
            data=150.0,
            grad=0,
            children=(
                Value(
                    data=15.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                )
            ),
            local_grads=(10.0, 15.0)
        )
        """

        return self * other

    def __truediv__(self, other):
        """
        División en Python.
        Para que el lenguaje sepa que hacer cuando se encuentra con
        val3 = val1/val2

        Ejemplos:
        ---------

        Inputs:
        val1 = Value(data=10.0)
        val2 = Value(data=15.0)
        val2/val1

        Outputs:
        Value(
            data=0.6666666666666666,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=0.06666666666666667,
                    grad=0,
                    children=(
                        Value(
                            data=15.0,
                            grad=0,
                            children=(),
                            local_grads=()
                        ),
                    ),
                    local_grads=(-0.0044444444444444444,)
                )
            ),
            local_grads=(0.06666666666666667, 10.0)
        )

        """
        return self * other**-1

    def __rtruediv__(self, other):
        """
        División en Python.
        Para que el lenguaje sepa que hacer cuando se encuentra con
        val3 = val2/val1

        Ejemplos:
        ---------
        Value(
            data=0.6666666666666666,
            grad=0,
            children=(
                Value(
                    data=10.0,
                    grad=0,
                    children=(),
                    local_grads=()
                ),
                Value(
                    data=0.06666666666666667,
                    grad=0,
                    children=(
                        Value(
                            data=15.0,
                            grad=0,
                            children=(),
                            local_grads=()
                        ),
                    ),
                    local_grads=(-0.0044444444444444444,)
                )
            ),
            local_grads=(0.06666666666666667, 10.0)
        )
        """

        return other * self**-1

    def backward(self):
        """
        Con esta función construimos el grafo de computación y calculamos los gradients
        de cada nodo.

        Lo hacemos en 2 pasos: por un lado, usamos la función de build_topo para
        recorrer todo el grafo y añadir todos y cada uno de los children.

        En un segundo caso, recorremos el grafo de computación en orden inverso y
        calculamos los gradients usando la Chain Rule.

        Chain Rule:
        -----------
        child.grad += local_grad * v.grad.
        𝜕L/𝜕child  += 𝜕L/𝜕v * 𝜕v/𝜕child
        """
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
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# La profunfidad de nuestra Red Neuronal, tiene 1 única capa
n_layer = 1

# Nr de dimensiones  de nuestro embedding
n_embd = 16

# El tamaña del bloque de atención es 16, porque el nombre más largo es de 15 caracteres
block_size = 16

# number of attention heads
n_head = 4

# derived dimension of each head
head_dim = n_embd // n_head


def matrix(
    *, nout: int, nin: int, matrix_name=str, std: float = 0.08
) -> list[list[Value]]:
    matrix_ = [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    print(f"El tamaño de nuestra matriz {matrix_name} es ({nout}, {nin})")
    return matrix_


# word token embedding
wte = matrix(nout=vocab_size, nin=n_embd, matrix_name="wte")

# work/weight position embedding
wpe = matrix(nout=block_size, nin=n_embd, matrix_name="wpe")

lm_head = matrix(nout=vocab_size, nin=n_embd, matrix_name="lm_head")

state_dict = {
    "wte": wte,
    "wpe": wpe,
    "lm_head": lm_head,
}

for i in range(n_layer):

    layer_ = f"layer{i}.attn_wq"
    state_dict[layer_] = matrix(nout=n_embd, nin=n_embd, matrix_name=layer_)

    # los attn_wk and attn_wv forman parte del kv_cache
    # sirven para saber que tokens anteriores son importantes
    # para un determinado token
    # attention work key
    layer_ = f"layer{i}.attn_wk"
    state_dict[layer_] = matrix(nout=n_embd, nin=n_embd, matrix_name=layer_)

    # attention work value
    layer_ = f"layer{i}.attn_wv"
    state_dict[layer_] = matrix(nout=n_embd, nin=n_embd, matrix_name=layer_)

    layer_ = f"layer{i}.attn_wo"
    state_dict[layer_] = matrix(nout=n_embd, nin=n_embd, matrix_name=layer_)

    layer_ = f"layer{i}.mlp_fc1"
    state_dict[layer_] = matrix(nout=4 * n_embd, nin=n_embd, matrix_name=layer_)

    layer_ = f"layer{i}.mlp_fc2"
    state_dict[layer_] = matrix(nout=n_embd, nin=4 * n_embd, matrix_name=layer_)

# flatten params into a single list[Value]
params = [p for mat in state_dict.values() for row in mat for p in row]


# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]  # token embedding
    pos_emb = state_dict["wpe"][pos_id]  # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
    x = rmsnorm(
        x
    )  # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # first moment buffer
v = [0.0] * len(params)  # second moment buffer

# Repeat in sequence
num_steps = 1000  # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(
        losses
    )  # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")

# Inference: may the model babble back to us
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([logit_ / temperature for logit_ in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
