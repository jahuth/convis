
Known Issues
============



 * `thano` `ScanOp`s contain a computational subgraph that is not connected to the outer graph
     - because of this, variables that are defined in the function supplied to the `ScanOp` (as opposed to supplied as an argument to it) are not accessible to the model
     - also the inner graphs are not plotted
     - **Remedy:** 
         + either treat inner graphs as distinct graphs in each graph wrapper (going from 1 to multiple graphs or add GraphWrappers which are then referenced)
         + or rewrite all graph exploration functions to also include `ScanOp` inner graphs
             * in this case the mapping between supplied arguments and copies has to be resolved!

 * there is no convenience class to create a 1d convolution exponential filter or 2d gauss filter

 * I can either ensure that labeling will be preserved when copying (by overriding the copying functions) OR make the models compile
     - I could revert the copying behaviour once the model is about to compile.

 * output_info of scan still does what it wants and optimizes similar zero shapes away:
     - "ERROR (theano.gof.opt): Optimization failure due to: remove_constants_and_unused_inputs_scan"
 * Alternatively I could rewrite all the labeling to use a lookup table instead. Then I only need to identify the variables eg. by name.