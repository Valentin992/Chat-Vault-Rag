# Bias-Variance Tradeoff

## Descomposición del error

El error de predicción esperado de un modelo se puede descomponer en tres componentes:

    Error = Bias² + Variance + Ruido irreducible

### Bias (sesgo)
Error sistemático: qué tan lejos está la predicción promedio del modelo de la realidad.
Un modelo con bias alto tiene suposiciones demasiado simples → underfitting.
**Ejemplo:** ajustar una línea recta a datos con forma de parábola.

### Variance
Sensibilidad a fluctuaciones en los datos de entrenamiento. Un modelo con variance alta
aprende el ruido específico del training set → overfitting.
**Ejemplo:** polinomio de grado 15 ajustado a 20 puntos — pasa exactamente por todos
pero falla en puntos nuevos.

### Ruido irreducible
Variabilidad inherente en los datos (medición imperfecta, procesos estocásticos).
No se puede eliminar con ningún modelo. Marca el piso teórico del error.

## El tradeoff

- **Modelos simples** (regresión lineal, árbol poco profundo):
  Bias alto, Variance baja → underfitting.
- **Modelos complejos** (red neuronal grande, árbol profundo):
  Bias bajo, Variance alta → overfitting.

No existe un modelo con Bias=0 y Variance=0 simultáneamente (salvo para problemas triviales).

## Fenómeno de doble descenso

Descubierto en deep learning moderno: si el modelo es suficientemente sobreparametrizado
(muchos más parámetros que datos de entrenamiento), el error de test vuelve a bajar
en lugar de subir indefinidamente.

Los LLMs con billones de parámetros operan en este régimen de interpolación. El tradeoff
clásico no aplica directamente en este extremo del espectro.

## Implicaciones prácticas

- Para reducir bias: usar modelo más complejo, más features, menos regularización.
- Para reducir variance: más datos, más regularización (dropout, L2), ensemble de modelos.
- **Cross-validación** es la herramienta práctica para estimar el error generalizable
  y diagnosticar si el problema es bias o variance.

## Relación con la regularización

La regularización (L1, L2, dropout) introduce bias deliberado para reducir variance.
El hiperparámetro de regularización controla dónde en el tradeoff te posicionas.
