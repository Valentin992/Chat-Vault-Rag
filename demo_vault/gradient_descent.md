# Descenso de Gradiente

El descenso de gradiente (gradient descent) es el algoritmo de optimización central en ML.
El objetivo es minimizar una función de pérdida L(θ) ajustando los parámetros θ del modelo.

## Intuición

Imagina que estás en una montaña con niebla y quieres bajar al valle. No puedes ver el valle
completo, solo el suelo justo bajo tus pies. La estrategia: mira la pendiente local y da un
paso en la dirección cuesta abajo. Repite hasta llegar al fondo.

El gradiente ∇L(θ) apunta en la dirección de mayor ascenso. Para bajar, nos movemos en la
dirección contraria:

    θ ← θ - η · ∇L(θ)

donde η (eta) es el **learning rate** (tasa de aprendizaje).

## Variantes

### Batch Gradient Descent
Calcula el gradiente sobre TODO el dataset. Exacto pero lento para datasets grandes.

### Stochastic Gradient Descent (SGD)
Calcula el gradiente de UN solo ejemplo por paso. Ruidoso pero rápido.
El ruido a veces ayuda a escapar de mínimos locales.

### Mini-batch SGD
El estándar en la práctica. Calcula el gradiente sobre un lote (batch) de 32-512 ejemplos.
Equilibrio entre precisión del gradiente y velocidad.

## Learning Rate

El hiperparámetro más crítico:
- Demasiado alto: diverge, la pérdida oscila o explota.
- Demasiado bajo: converge muy lento, se queda atascado.
- Solución práctica: learning rate schedules (warm-up + decay) o optimizadores adaptativos.

## Optimizadores modernos

En la práctica casi nunca se usa SGD puro. Los optimizadores adaptativos ajustan el LR
por parámetro automáticamente:
- **Adam**: combina momentum + RMSprop. Default razonable para la mayoría de tareas.
- **AdamW**: Adam + weight decay desacoplado. Mejor para transformers.
- **SGD + momentum**: sigue siendo competitivo en visión (ResNets).

## Mínimos locales vs. globales

En superficies de pérdida no convexas (redes neuronales) existen múltiples mínimos locales.
En la práctica, con redes grandes y datos suficientes, la mayoría de mínimos locales tienen
calidad similar al global — el ruido del SGD ayuda a escapar de los malos.
