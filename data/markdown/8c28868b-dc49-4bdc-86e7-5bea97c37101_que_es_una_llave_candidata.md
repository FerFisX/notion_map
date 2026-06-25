que es una llave candidata

it_bambootec

Owner

Tags

Created time

Una llave candidata (o clave candidata) es un conjunto de una o más columnas
en una tabla de base de datos que puede identificar de manera única cada fila de
la tabla. Es importante que una llave candidata cumpla dos propiedades clave:

  Unicidad Los valores combinados de las columnas que forman la llave

candidata deben ser únicos para cada fila en la tabla. Esto asegura que no
haya duplicados en las filas.

  Minimalidad La llave candidata debe ser el conjunto mínimo de columnas

necesario para garantizar la unicidad. Esto significa que no se puede eliminar
ninguna columna de la llave candidata sin perder la capacidad de identificar

de manera única cada fila.

Una tabla puede tener varias llaves candidatas. De estas, una se selecciona como

la llave primaria (primary key), que es la que se usa principalmente para
identificar de manera única las filas. Las llaves candidatas que no son
seleccionadas como llaves primarias pueden seguir siendo útiles como llaves

alternativas.

Ejemplo:

En una tabla de empleados:

DNI  y  Correo Electrónico  pueden ser llaves candidatas, ya que ambos pueden

identificar de manera única a un empleado.

De estas dos llaves candidatas, se elige una como la llave primaria (por

ejemplo,  DNI ).

October 24, 2024 1057 AMque es una llave candidata1