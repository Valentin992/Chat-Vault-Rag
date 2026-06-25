# Backpropagation

## Qué es

Backpropagation es el algoritmo para calcular los gradientes de la función de pérdida
respecto a cada peso de la red. Es la regla de la cadena (chain rule) del cálculo diferencial
aplicada sistemáticamente sobre el grafo computacional de la red.

## La regla de la cadena

Si L depende de z que depende de w:

    ∂L/∂w = (∂L/∂z) · (∂z/∂w)

En una red de N capas, el gradiente de la pérdida respecto a los pesos de la capa i
se calcula propagando hacia atrás los gradientes desde la capa de salida hasta esa capa.

## Forward pass → Backward pass

1. **Forward pass**: calcular la salida y guardar las activaciones intermedias.
   Las activaciones intermedias son necesarias para el backward (se guarda el grafo).
2. **Backward pass**: empezando desde ∂L/∂output, propagar hacia atrás usando chain rule.
   Para cada capa: calcular ∂L/∂w (para actualizar pesos) y ∂L/∂input (para seguir propagando).

## Vanishing Gradient Problem

En redes profundas, los gradientes se multiplican en cada capa. Si |∂z/∂w| < 1
consistentemente, el producto de muchas capas → 0. Las primeras capas dejan de aprender.

**Causas:**
- Sigmoid/tanh: gradiente máximo 0.25 → colapsa rápido en redes profundas.
- Redes muy profundas sin conexiones residuales.

**Soluciones:**
- **ReLU**: gradiente 1 para activaciones positivas (no se desvanece).
- **Conexiones residuales** (ResNets): el gradiente tiene un "camino corto" hasta las primeras capas.
- **Batch normalization**: estabiliza las distribuciones de activación entre capas.
- **Inicialización cuidadosa**: Xavier (tanh), He (ReLU) escalan los pesos iniciales por capa.

## Exploding Gradient

El problema inverso: gradientes que crecen exponencialmente → NaN, divergencia.
**Solución estándar:** gradient clipping — si ||g|| > threshold, g ← g · (threshold / ||g||).
Uso común en RNNs y entrenamiento de LLMs.

## Automatic Differentiation (autograd)

En la práctica, nadie implementa backprop a mano. Frameworks como PyTorch construyen
un grafo computacional dinámico durante el forward pass y calculan los gradientes
automáticamente con `.backward()`. Cada operación registra su función derivada;
autograd recorre el grafo en orden inverso (topological sort).
