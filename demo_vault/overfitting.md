# Overfitting y Regularización

## Qué es overfitting

Un modelo hace overfitting cuando memoriza los datos de entrenamiento en lugar de aprender
patrones generalizables. Resultado: baja pérdida en train, alta pérdida en test (mala generalización).

## Señales de overfitting

- La loss de train sigue bajando pero la de validación sube o se estanca.
- El modelo funciona perfecto en entrenamiento pero mal en datos nuevos.
- Las predicciones son muy confiadas (probabilidades extremas) para ejemplos fuera de distribución.

## Técnicas de regularización

### L2 (Weight Decay)
Añade penalización ||w||² a la función de pérdida. Empuja los pesos hacia cero,
distribuyendo la "energía" del modelo entre más features. Equivale a un prior Gaussiano
sobre los pesos.

    L_total = L_task + λ · ||w||²

### L1 (Lasso)
Penalización ||w||₁. Induce sparsity: algunos pesos llegan exactamente a 0
(útil para selección de features).

### Dropout
Durante el entrenamiento, apaga neuronas aleatoriamente con probabilidad p.
Fuerza al modelo a aprender representaciones redundantes y robustas.
En inferencia, todos los nodos están activos (escalados por 1-p).

### Data Augmentation
Generar nuevos ejemplos de entrenamiento aplicando transformaciones (rotaciones, crops, ruido).
Aumenta el tamaño efectivo del dataset sin más datos reales.

### Early Stopping
Monitorizar la pérdida de validación y detener el entrenamiento cuando empieza a subir.
Requiere un split de validación separado.

### Batch Normalization
Normaliza las activaciones dentro de cada mini-batch. Tiene efecto regularizador implícito
(introduce ruido en las estadísticas de batch).

## Cuándo aplicar qué

- Modelo pequeño + dataset grande: raramente overfitting. No regularizar agresivo.
- Modelo grande + dataset pequeño: dropout + weight decay + data augmentation.
- Siempre monitorizar la curva train/val loss.

## Underfitting

El problema contrario: el modelo es demasiado simple para capturar el patrón. La loss de train
también es alta. Soluciones: modelo más complejo, más features, menos regularización.
