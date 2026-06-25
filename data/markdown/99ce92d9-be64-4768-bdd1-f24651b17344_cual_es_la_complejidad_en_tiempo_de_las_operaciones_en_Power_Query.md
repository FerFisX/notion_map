cual es la complejidad en tiempo
de las operaciones en Power
Query

Adrian Villarroel

Complejidad

Notación Big-O

Descripción

Owner

Tags

Created time

Operación

Acceso directo a

columna

Selección de

columnas

O1

O1

Renombrar columnas

O1

Eliminación de

columnas

Filtro de filas

(condicional)

Combinación de

columnas

Transformación de

datos

O1

O(n)

O(n)

O(n)

Ordenación de filas

O(n log n)

Acceso a un valor específico en una

columna sin realizar cálculos adicionales.

Selecciona un subconjunto de columnas

sin realizar transformaciones.

Cambia el nombre de una columna sin

modificar el contenido.

Simplemente elimina referencias a las

columnas; no afecta el tamaño de los

datos.

Filtra filas según una condición sin

transformar los valores.

Combina dos o más columnas; su

complejidad depende del tamaño de las

columnas.

Aplicación de funciones sobre las

columnas de datos, como cambiar el tipo

de dato.

Ordenar una tabla por una o más

columnas.

Eliminación de

duplicados

O(n log n)

Implica ordenar los datos antes de eliminar

los duplicados.

October 14, 2024 838 AMcual es la complejidad en tiempo de las operaciones en Power Query1Group By
Agrupación)

O(n log n)

Requiere ordenar y agrupar según uno o
más criterios.

Merge/Join (unión de
tablas)

O(n log n)  O(n^2)

La complejidad depende del tipo de unión
(inner, outer, etc.) y el tamaño de los datos.

Pivot Pivoteo)

O(n log n)  O(n^2)

Unpivot Despivot)

O(n log n)  O(n^2)

Expansión de tablas
anidadas

Aplicar funciones
personalizadas

O(n^2)

O(n^2)

Transforma las columnas en filas o
viceversa, reorganizando los datos.

Convierte columnas en filas; es costoso
cuando se realiza sobre grandes datasets.

Expande datos dentro de columnas que
contienen tablas o listas anidadas.

Funciones personalizadas pueden ser
costosas si no están optimizadas.

cual es la complejidad en tiempo de las operaciones en Power Query2