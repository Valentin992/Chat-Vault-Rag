# Bases de Datos Vectoriales

## Qué son

Una base de datos vectorial almacena embeddings (vectores de alta dimensionalidad) y permite
búsqueda eficiente por similitud semántica. En lugar de buscar por valor exacto (SQL),
se buscan los k vectores más cercanos a un vector consulta (nearest neighbor search).

## Por qué no basta con SQL

Para vectores de 1536 dimensiones, comparar contra millones de vectores con fuerza bruta
es O(n · d) — inviable a escala. Las vector DBs usan índices aproximados (ANN: Approximate
Nearest Neighbor) que intercambian algo de precisión por velocidad exponencialmente mayor.

## Algoritmos ANN

### HNSW (Hierarchical Navigable Small World)
El estándar actual. Construye un grafo de navegación multi-capa:
- Capas superiores: pocas conexiones de largo alcance (navegación rápida hacia la zona correcta).
- Capas inferiores: más conexiones de corto alcance (refinamiento local).

Búsqueda: entry point → navegar el grafo reduciendo distancia al query en cada paso.
Trade-off: `ef_construction` controla calidad vs. velocidad de construcción del índice;
`ef_search` controla calidad vs. velocidad de búsqueda.

### IVF (Inverted File Index)
Divide el espacio en clusters (k-means). En búsqueda, examina solo los n_probe clusters
más cercanos. Más rápido que HNSW en inserción masiva; menos preciso.

## Métricas de distancia

- **Coseno**: mide ángulo entre vectores. Independiente de la magnitud. Default para texto.
- **Producto punto**: equivale a coseno si los vectores están normalizados a longitud 1.
- **L2 (euclídea)**: distancia absoluta. Útil cuando la magnitud del vector importa.

## Opciones populares

| DB | Notas |
|----|-------|
| **ChromaDB** | Fácil para prototipos, persistencia local, open-source. |
| **Pinecone** | Managed cloud, escala automático, baja latencia. |
| **Weaviate** | Self-hosted o cloud, multi-modal, GraphQL API. |
| **Qdrant** | Alta performance, filtros avanzados, open-source. |
| **pgvector** | Extensión PostgreSQL, para quienes ya usan Postgres. |

## Filtrado híbrido

Las vector DBs modernas permiten combinar búsqueda semántica + filtros exactos sobre metadata:
"Dame los 5 chunks más similares a esta query, donde fuente='capitulo_3.md'".
Esencial para RAG sobre documentos heterogéneos con múltiples categorías o fechas.
