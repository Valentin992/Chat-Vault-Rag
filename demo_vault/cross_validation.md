# Validación Cruzada (Cross-Validation)

## El problema

¿Cómo evaluar el rendimiento generalizable de un modelo con datos limitados?

Usar todo el dataset para entrenar y medir el error de train subestima el error real
(el modelo se especializa en los datos de entrenamiento). Necesitamos una estimación
honesta del error en datos no vistos.

## Train / Validation / Test Split

La separación estándar:
- **Train (~70%)**: entrenar el modelo.
- **Validation (~15%)**: seleccionar hiperparámetros y arquitectura.
- **Test (~15%)**: evaluación final honesta. NUNCA se usa para tomar decisiones de modelado.

Si el test set influye en alguna decisión de diseño, ya no es una estimación honesta.

## K-Fold Cross-Validation

Cuando los datos son escasos, un único split puede ser muy ruidoso.

**Proceso:**
1. Dividir los datos en K subconjuntos (folds) iguales.
2. Para i = 1..K: entrenar con los K-1 folds restantes, evaluar en el fold i.
3. Promediar los K scores de evaluación.

**Resultado:** estimación más estable del error generalizable, con todos los datos
usados para entrenar al menos una vez.
**K típico:** 5 o 10. Leave-One-Out (K=n) para datasets muy pequeños.

## Stratified K-Fold

En clasificación desbalanceada, el K-fold estándar puede generar folds con distribuciones
de clase muy distintas. Stratified K-fold garantiza que cada fold mantiene la proporción
original de clases.

## Nested Cross-Validation

Para selección de hiperparámetros + evaluación simultánea sin contaminación:
- **Loop externo**: estima el error generalizable del pipeline completo.
- **Loop interno**: selecciona los mejores hiperparámetros dentro de cada fold externo.

Computacionalmente costoso pero estadísticamente honesto.
Evita el sesgo de selección ("test set leakage").

## Cross-Validation en Time Series

Con datos temporales, NO se puede hacer CV estándar (el futuro no puede estar en train).
Usar **TimeSeriesSplit**: el fold de test siempre es temporalmente posterior al de train.

## Cuándo usar qué

- Dataset grande (>10k ejemplos): un único split suele ser suficiente.
- Dataset mediano: K-Fold con K=5 o K=10.
- Dataset pequeño (<500): Leave-One-Out o K-Fold con K grande.
- Time series: TimeSeriesSplit siempre.
