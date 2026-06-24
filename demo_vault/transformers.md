# Transformers y Atención

## El problema que resuelven

Las RNNs procesan secuencias token a token. Problemas:
- Dependencias de largo alcance: el gradiente se desvanece en secuencias largas.
- Secuencial: no paralelizable durante el entrenamiento.

Los transformers (Vaswani et al., 2017, "Attention Is All You Need") eliminan la recurrencia.

## Mecanismo de Self-Attention

Para cada token, la atención decide cuánto "fijarse" en cada otro token de la secuencia.

**Queries, Keys, Values:**
Cada token genera tres vectores: Q (lo que busca), K (lo que ofrece), V (su contenido).

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

- QKᵀ: producto punto → score de relevancia entre pares de tokens.
- √d_k: factor de escala para evitar gradientes pequeños con dimensiones altas.
- softmax: normaliza los scores a distribución de probabilidad.
- · V: suma ponderada de los valores según los scores.

## Multi-Head Attention

En lugar de una sola atención, el transformer usa h cabezas en paralelo, cada una con
sus propias matrices W_Q, W_K, W_V. Cada cabeza aprende un tipo de relación distinto
(sintáctica, semántica, correferencia, etc.). Los outputs se concatenan y proyectan.

## Arquitectura

```
Input → Embedding + Positional Encoding
  ↓
[Encoder Block] × N:
  Multi-Head Self-Attention → Add & Norm
  Feed-Forward Network → Add & Norm
```

**Positional encoding:** Los transformers no tienen noción de orden inherente (diferente a RNNs).
Se añade una señal posicional al embedding de cada token para que el modelo sepa la posición.

**Add & Norm:** Conexión residual (suma la entrada con la salida del sub-bloque) +
Layer Normalization. Esencial para entrenar redes profundas sin vanishing gradient.

## Variantes principales

- **BERT**: solo encoder. Bidireccional. Pre-training con masked language modeling. Bueno para clasificación.
- **GPT**: solo decoder. Autoregresivo (predice el siguiente token). Base de los LLMs actuales.
- **T5**: encoder-decoder. Trata todo como seq2seq (traducción, resumen).
- **LLaMA, Mistral**: decoder-only. Base de los modelos open-source modernos.

## Por qué dominan

Paralelización total durante el entrenamiento + atención global sobre toda la secuencia
→ mejor captura de contexto + escalado eficiente con más datos y parámetros (scaling laws).
