from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="mrm8488/bert2bert_shared-spanish-finetuned-summarization"
)

texto = """
Las ventas aumentaron un 20% en el último trimestre.
El producto más vendido fue el modelo X200.
La región con mayor crecimiento fue Latinoamérica.
"""

resumen = summarizer(texto, max_length=40)

print(resumen[0]["summary_text"])