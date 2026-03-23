from transformers import pipeline

classifier = pipeline("sentiment-analysis")

resultado = classifier(
    #"Terrible service and bad products"
    "Products are good"
)

print(resultado)