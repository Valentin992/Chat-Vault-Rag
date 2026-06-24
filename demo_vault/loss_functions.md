# Funciones de Pérdida

La función de pérdida (loss function) mide qué tan equivocado está el modelo.
El entrenamiento minimiza esta función vía descenso de gradiente.

## Regresión

### MSE — Mean Squared Error
    L = (1/n) Σ (yᵢ - ŷᵢ)²
Penaliza errores grandes mucho más que errores pequeños (efecto cuadrático).
Sensible a outliers. Default para regresión cuando los outliers no son un problema.

### MAE — Mean Absolute Error
    L = (1/n) Σ |yᵢ - ŷᵢ|
Menos sensible a outliers que MSE. Gradiente constante (±1/n), no diferenciable en 0.

### Huber Loss
Combina MSE (para errores pequeños) y MAE (para errores grandes).
Robusto a outliers con gradientes suaves cerca de 0. Controlado por parámetro δ.

## Clasificación

### Binary Cross-Entropy (Log Loss)
Para clasificación binaria. Salida: probabilidad p ∈ (0,1) via sigmoid.
    L = -[y log(p) + (1-y) log(1-p)]
Si y=1: penaliza log(p) → si p≈0, pérdida enorme.
Si y=0: penaliza log(1-p) → si p≈1, pérdida enorme.

### Categorical Cross-Entropy
Para clasificación multi-clase. Salida: distribución softmax.
    L = -Σₖ yₖ log(pₖ)
Con targets one-hot: L = -log(p_clase_correcta).
Interpretación: maximizar la log-probabilidad de la clase correcta.

## Modelos de lenguaje

### Cross-Entropy sobre el vocabulario
Los LLMs minimizan la cross-entropy de la distribución sobre el vocabulario completo
en cada posición de la secuencia.

**Perplexity** = exp(cross-entropy promedio).
Interpretación: "¿sobre cuántas palabras equivalente está dudando el modelo por posición?"
Un modelo perfecto: perplexity=1. Un modelo aleatorio: perplexity=|vocab| (~50k).
GPT-4 tiene perplexity ~2-5 en benchmarks estándar.

## Consideraciones

- La elección de la función de pérdida debe reflejar el objetivo real, no solo lo conveniente.
- En clasificación desbalanceada: pérdidas pesadas (weighted cross-entropy) o focal loss.
- Una pérdida bien elegida es parte del inductive bias del modelo.
