# Redes Neuronales

## Neurona artificial

Una neurona calcula:

    y = f(w · x + b)

- **x**: vector de entrada.
- **w**: pesos (weights) — lo que el modelo aprende.
- **b**: bias — desplazamiento que permite al modelo ajustarse independientemente de x.
- **f**: función de activación — introduce no-linealidad.

Sin activación no-lineal, una red profunda colapsa a una transformación lineal simple.

## Funciones de activación

### ReLU (Rectified Linear Unit)
    f(x) = max(0, x)
Default en capas ocultas. Simple, evita el vanishing gradient problem para activaciones
positivas. Problema: neuronas que reciben entradas negativas tienen gradiente 0 y
pueden "morir" (siempre apagadas).

### Sigmoid
    f(x) = 1 / (1 + e^(-x))
Salida entre 0 y 1. Usada en la capa de salida para clasificación binaria.
Problema: gradientes muy pequeños en las colas → vanishing gradient.

### Softmax
    f(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)
Distribución de probabilidad sobre k clases. Salida de clasificación multi-clase.

### GeLU / SiLU
Variantes suavizadas de ReLU que dominan en transformers modernos (GPT, BERT).

## Arquitecturas

### Fully Connected (Dense / MLP)
Cada neurona conectada a todas las de la capa anterior. Muy flexible, muchos parámetros.
Base de los transformers en sus capas feed-forward.

### Convolucional (CNN)
Filtros compartidos que detectan patrones locales. Eficiente para imágenes.
Invarianza traslacional: el mismo filtro detecta un borde en cualquier posición.

### Recurrente (RNN, LSTM, GRU)
Procesan secuencias manteniendo estado oculto. Superadas en NLP por los transformers.
LSTM resuelve el vanishing gradient con gating explícito (forget, input, output gates).

## Universal Approximation Theorem

Una red con al menos una capa oculta y suficientes neuronas puede aproximar cualquier
función continua arbitrariamente bien. Esto es teórico — en la práctica, la profundidad
importa más que la anchura para capturar jerarquías de features.

## Forward Pass vs. Backward Pass

- **Forward pass**: calcular la predicción ŷ y la pérdida L.
- **Backward pass**: calcular ∂L/∂w para cada peso usando backpropagation.
- **Actualización**: aplicar los gradientes con el optimizador (Adam, SGD+momentum).
