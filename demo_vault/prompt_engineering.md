# Prompt Engineering

## Qué es

El arte y la ciencia de diseñar entradas (prompts) para obtener las mejores salidas de un LLM.
Los modelos son sensibles a la formulación, el orden y la estructura del input.

## Técnicas principales

### Zero-shot
Describir la tarea directamente, sin ejemplos.
```
Clasifica el sentimiento: "El producto llegó tarde y roto."
Responde SOLO: Positivo / Negativo / Neutro.
```
Funciona para tareas que el modelo conoce bien de su pre-training.

### Few-shot
Incluir 2-5 ejemplos de input/output en el prompt antes del caso real.
Muy efectivo: muestra el formato esperado y calibra el estilo de respuesta.

### Chain-of-Thought (CoT)
Instruir al modelo a "pensar paso a paso" antes de dar la respuesta final.
Mejora dramáticamente el razonamiento en problemas matemáticos y lógicos.
Variante: "think step by step" en el prompt sin ejemplos (zero-shot CoT).

### ReAct (Reason + Act)
El modelo alterna entre razonamiento ("Thought") y acciones ("Action") con herramientas
externas. Base de muchos sistemas agentic actuales.

## Principios generales

1. **Ser específico**: "Responde en máximo 3 oraciones" > "Sé conciso".
2. **Formato explícito**: indicar el formato de salida esperado (JSON, lista, tabla).
3. **Role prompting**: "Eres un experto en seguridad" puede mejorar respuestas técnicas.
4. **Separar instrucciones de datos**: usar delimitadores claros (```, etiquetas XML).
5. **Instrucciones afirmativas**: "Di lo que SÍ debes hacer" > "No hagas X".

## Prompt Caching

Los LLMs modernos (Claude, GPT-4) permiten cachear prefijos de prompt para reducir
latencia y costo en conversaciones multi-turno. El system prompt (estático) es el
mejor candidato: si no cambia entre turnos, se cachea automáticamente.

Consecuencia práctica: mantener el system prompt 100% estático (sin fechas ni variables
dinámicas) para maximizar el hit rate del caché de Anthropic.

## Limitaciones

- **Prompt injection**: un usuario malicioso puede incluir instrucciones en sus inputs
  para sobreescribir el system prompt. Mitigación: separación clara instrucciones/datos.
- **Sensibilidad a la formulación**: pequeños cambios en el prompt pueden cambiar
  significativamente la salida — necesario evaluar sistemáticamente (evals).
- **Instrucciones en conflicto**: el modelo puede priorizar instrucciones del usuario
  sobre el system prompt. Ser explícito sobre la jerarquía de instrucciones.
