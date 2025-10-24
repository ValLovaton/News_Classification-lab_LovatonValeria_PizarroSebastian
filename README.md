# Task 2 — Transformer News Classification (AG News + Bonus RPP)

Este repositorio desarrolla y evalúa modelos *Transformer* aplicados a la clasificación automática de noticias, utilizando el dataset AG News y un conjunto adicional de 50 noticias reales de RPP Perú como extensión práctica (*Bonus Task*).

---

## Objetivo General

Entrenar, comparar y analizar el desempeño de modelos basados en *Transformers* (RoBERTa, DeBERTa y BERT Multilingual) en la clasificación de noticias según cuatro categorías:
**0 - World | 1 - Sports | 2 - Business | 3 - Science/Technology.**

Además, contrastar los resultados de los modelos con las etiquetas generadas por un LLM aplicado sobre las noticias de RPP, para evaluar coherencia y diferencias semánticas.

---

## Requisitos

* Python ≥ 3.10
* Librerías principales:
  `transformers`, `datasets`, `torch`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`
* Instalación:

  ```bash
  pip install -r requirements.txt
  ```

---

## Estructura del Repositorio

```
├── data/
│   ├── rpp_source.json               ← Noticias de RPP clasificadas
│
├── notebooks/
│   ├── agnews_train_eval.ipynb       ← Carga y preprocesamiento del dataset AG News
│   ├── agnews_train_eval_step2_roberta.ipynb  ← Entrenamiento RoBERTa
│   ├── agnews_train_eval_step3_compare.ipynb  ← Comparación de resultados
│   ├── bonus_prepare_rpp.ipynb       ← Clasificación LLM y comparación (Bonus Task)
│
├── outputs/                          ← Resultados y gráficos generados
├── src/                              ← Módulos auxiliares (entrenamiento, métricas, etc.)
├── requirements.txt
└── README.md
```

---

## 🚀 Pipeline de Entrenamiento y Evaluación

###  Carga y Preparación de Datos

* Dataset: `load_dataset("ag_news")`
* División:

  * 70% entrenamiento
  * 15% validación
  * 15% prueba (reservado para evaluación final)

### Modelos Entrenados

Se entrenaron tres modelos base en Colab, cada uno con *Hugging Face Transformers*:

| Modelo               | Checkpoint                     | Épocas | Contexto   |
| -------------------- | ------------------------------ | ------ | ---------- |
| **RoBERTa**          | `roberta-base`                 | 1      | 512 tokens |
| **DeBERTa**          | `microsoft/deberta-v3-base`    | 1      | 512 tokens |
| **BERT Multilingüe** | `bert-base-multilingual-cased` | 1      | 512 tokens |

---

## Resultados de Desempeño (AG News)

| Modelo                           | F1-score   |
| -------------------------------- | ---------- |
| **roberta-base**                 | 0.7037     |
| **microsoft/deberta-v3-base**    | 0.2917     |
| **bert-base-multilingual-cased** | **0.7833** |

📈 Comparación visual:

![F1-score Comparison](outputs/f1_scores_comparison.png)

El modelo **BERT Multilingüe** obtuvo el mejor desempeño, con un **F1 de 0.78**, superando tanto a RoBERTa (0.70) como a DeBERTa (0.29). Esto sugiere que la cobertura lingüística multilingüe ayudó a una mejor comprensión contextual en el dataset.

---

## Bonus Task — Clasificación de Noticias RPP (LLM vs Transformers)

### **Flujo**

1. Se usaron las 50 noticias más recientes del RSS de RPP Perú.
2. Un LLM (ChatGPT API) clasificó cada noticia según las categorías AG News.
3. Los tres modelos entrenados aplicaron inferencia sobre las mismas noticias.
4. Se compararon los resultados (predicciones vs LLM) mediante F1-score.

### **Resultados**

| Comparación          | Observaciones                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **BERT Multilingüe** | Mostró mayor coherencia con las etiquetas del LLM. Capta mejor el lenguaje periodístico y regional. |
| **RoBERTa**          | Aceptable, aunque tiende a confundir temas de ciencia y negocios.                                   |
| **DeBERTa**          | Peor desempeño: bajo ajuste en textos fuera del dominio original inglés.                            |

**Conclusión del Bonus:**
El modelo BERT Multilingüe logra la mayor consistencia con las clasificaciones del LLM, lo que confirma su capacidad para manejar contextos y estilos de texto no estrictamente anglosajones, como las noticias peruanas.

---

## Métricas Principales

* **Métrica usada:** Macro F1-score (balancea clases desiguales)
* **Batch size:** 8
* **Optimizer:** AdamW
* **Learning rate:** 5e-5
* **Seed reproducible:** 42
* **Checkpoints guardados:** en `/outputs`
---

## Discusión e Interpretación

El desempeño diferencial entre los modelos se explica por:

* **Dominio del pre-entrenamiento:** RoBERTa y DeBERTa fueron preentrenados en inglés, lo que reduce su comprensión semántica para textos con estructura mixta o hispana.
* **BERT Multilingüe** muestra una mejor generalización a contextos diversos, y por ello logra una clasificación más estable tanto en AG News como en RPP.
* Las divergencias con el LLM se deben a diferencias de contexto y longitudes de texto (los modelos locales procesan tokens truncados, mientras que el LLM conserva contexto completo).

---

## Reproducibilidad

Para ejecutar todo el pipeline en Colab:

```bash
!git clone https://github.com/ValLovaton/agnews-transformers-lab_LovatonValeria_PizarroSebastian.git
%cd agnews-transformers-lab_LovatonValeria_PizarroSebastian
!pip install -r requirements.txt
```

Abrir y correr secuencialmente los notebooks de `/notebooks/`:

1. `agnews_train_eval.ipynb`
2. `agnews_train_eval_step2_roberta.ipynb`
3. `agnews_train_eval_step3_compare.ipynb`
4. `bonus_prepare_rpp.ipynb` *(para el Bonus Task con RPP)*

---

## Conclusión

El proyecto demuestra que los modelos *Transformer* pueden adaptarse de manera efectiva a la clasificación de noticias, y que los modelos multilingües ofrecen una ventaja clara al aplicarse a fuentes en español.
La comparación con un LLM permitió validar la coherencia semántica de los resultados y resaltar las limitaciones de los modelos entrenados en dominios monolingües.


¿Quieres que te agregue también una **versión corta (resumen ejecutivo)** al final del README para defensa oral o repositorio público (como un párrafo resumen tipo abstract)?
