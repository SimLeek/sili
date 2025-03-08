# SILi
Sparse Intelligence Library

# ToDO

* Automatic execution graph generation
  * store the modules and dependency buffers inside buffer objects
  * Make a pipeline class that receives the output, loss, and optim buffers (if any)
    * having the pipeline called once makes keeping all computation on gpu until done easier