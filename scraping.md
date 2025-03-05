```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re  # Importamos re para expresiones regulares

# URL de Emol
url = "https://www.emol.com/"

# Obtener el contenido de la página
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Buscar elementos con clase que comience con "col_center_noticia"
titles = [title.text.strip() for title in soup.find_all(class_=re.compile("^col_center_noticia"))]

# Guardar en un DataFrame
df = pd.DataFrame({"Title": titles})

# Mostrar primeros resultados
print(df.head())

# Guardar en CSV
df.to_csv("news.csv", index=False)


```

                                                   Title
    0  1 \r\n    \n\r\n        Propuesta de compra y ...
    1  Trump vaticina el fin de la guerra en Ucrania:...
    2  Artista invitada, animadora y reina: La vasta ...
    3  Miguel "Negro" Piñera internado en clínica en ...
    4  Telefónica anuncia la venta de su filial argen...
    


```python

```
