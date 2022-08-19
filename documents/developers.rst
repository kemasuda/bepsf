=============================
Sphinx doc
=============================


To generate sphinx doc, run 

.. code:: python3
	  	  
   python setup.py install
   rm -rf documents/bepsf
   sphinx-apidoc -F -o documents/bepsf src/bepsf
   cd documents
   make clean
   make html
