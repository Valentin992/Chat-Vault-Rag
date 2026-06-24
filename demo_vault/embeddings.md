# Embeddings

## Definición

Un embedding es una representación densa de baja dimensionalidad de un objeto (token, frase,
imagen, usuario) en un espacio vectorial continuo. La clave: objetos semánticamente similares
tienen vectores cercanos en el espacio.

## Por qué no one-hot encoding

One-hot: vector de longitud |vocabulario| con un único 1. Problemas:
- Alta dimensionalidad (50k+ dimensiones para vocabularios de NLP).
- No captura similitud: "gato" y "felino" son ortogonales.
- Sin relaciones aritméticas: "rey" - "hombre" + "mujer" = nada.

Los embeddings resuelven todo esto en ~100-1536 dimensiones.

## Word2Vec (2013)

Primer embedding popular. Dos variantes:
- **CBOW**: predice la palabra central dadas las palabras de contexto.
- **Skip-gram**: predice las palabras de contexto dada la palabra central.

Propiedad famosa: king - man + woman ≈ queen. Los vectores capturan relaciones
semánticas y sintácticas como aritmética vectorial.

## Embeddings en redes neuronales

En transformers, la primera capa es una embedding table: matriz E de tamaño
(|vocab| × d_model). Cada token es un índice que selecciona una fila. Los pesos
se aprenden end-to-end con el resto del modelo.

## Sentence Embeddings

Los embeddings de palabras no capturan el significado de frases completas.
Modelos como Sentence-BERT o text-embedding-3-small (OpenAI) generan un único
vector para una frase o párrafo entero, capturando el significado global.

Uso típico: búsqueda semántica, clustering de documentos, RAG.

## Propiedades del espacio

- **Distancia coseno**: mide similitud de dirección, independiente de la magnitud.
  Preferida sobre distancia euclídea para embeddings de texto.
- **Dimensionalidad**: text-embedding-3-small usa 1536 dims.
  Más dimensiones ≠ siempre mejor; depende del modelo y la tarea.
- **Normalización**: muchos sistemas normalizan los vectores a longitud 1 →
  cosine similarity = producto punto.

## Modelos de embeddings populares

| Modelo | Dims | Notas |
|--------|------|-------|
| text-embedding-3-small | 1536 | OpenAI, barato (~$0.002/vault) |
| text-embedding-3-large | 3072 | OpenAI, mejor calidad |
| voyage-3 | 1024 | Anthropic recomienda Voyage AI |
| all-MiniLM-L6-v2 | 384 | Local, sin API, sentence-transformers |
