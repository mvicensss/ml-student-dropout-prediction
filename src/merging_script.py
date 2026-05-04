import nbformat

nb2 = nbformat.read('data/raw/Dataset_Analysis_Persona2.ipynb', as_version=4)
nb1 = nbformat.read('data/raw/Limpieza_de_Dataset.ipynb', as_version=4)

nb2.cells.extend(nb1.cells) # Append cells of the second notebook to the first one

nbformat.write(nb2, 'full_pipeline.ipynb') # Write the merged notebook to a new file