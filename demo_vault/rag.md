# RAG — Retrieval-Augmented Generation

## Motivación

Los LLMs tienen conocimiento limitado a su fecha de corte, no acceden a información privada,
y a veces alucinan (inventan hechos plausibles pero falsos).

**RAG** conecta el LLM a una base de conocimiento externa en tiempo de inferencia,
sin necesidad de re-entrenar el modelo.

## Pipeline

```
Pregunta del usuario
  ↓
1. Embed la pregunta (mismo modelo que se usó para el corpus)
  ↓
2. Búsqueda vectorial: recuperar los k chunks más similares de la vector DB
  ↓
3. Construir el prompt: [instrucción] + [chunks recuperados] + [pregunta]
  ↓
4. LLM genera la respuesta, citando las fuentes [n]
  ↓
Respuesta + citas trazables
```

## Componentes

### Indexación (offline)
- **Chunking**: dividir documentos en fragmentos (~500-1000 tokens).
  Decisiones: tamaño del chunk, overlap, respetar estructura (párrafos, headers).
- **Embedding**: convertir cada chunk en un vector con un modelo de embeddings.
- **Vector DB**: almacenar los vectores para búsqueda por similitud (Chroma, Pinecone, Weaviate).

### Retrieval (online)
- Embed la query.
- Buscar top-k chunks por similitud coseno.
- Trade-off: k grande → más contexto → más costo y latencia; k chico → puede perder hechos clave.

### Generación
- System prompt: "Responde SOLO usando el contexto. Si no está en el contexto, dilo."
- Anti-alucinación: la instrucción de abstención es crítica. Sin ella, el LLM extrapola.
- Citas numeradas [n]: cada afirmación trazable al chunk fuente.

## Métricas de evaluación

- **Recall@k**: ¿está la nota correcta en el top-k recuperado?
- **Groundedness**: ¿cada afirmación de la respuesta tiene soporte en el contexto?
- **Cobertura de hechos**: ¿cuántos hechos clave de la respuesta ideal están cubiertos?
- **Abstención**: ¿el sistema dice "no sé" correctamente para preguntas fuera de alcance?

## Limitaciones

- Calidad del chunking: chunks demasiado cortos pierden contexto; demasiado largos diluyen la señal.
- "Lost in the middle": los LLMs tienden a ignorar el contexto que está en el medio del prompt.
- Recall por fuente ≠ recall por hecho: recuperar la nota correcta no garantiza extraer
  el fragmento exacto con el hecho buscado.

## RAG vs Fine-tuning

| | RAG | Fine-tuning |
|--|-----|-------------|
| Datos nuevos | Inmediato (re-indexar) | Re-entrenar |
| Trazabilidad | Alta (citas) | Baja |
| Costo | Bajo (solo inferencia) | Alto (GPU) |
| Cuando usar | Conocimiento privado/actualizable | Estilo/formato/vocabulario específico |
